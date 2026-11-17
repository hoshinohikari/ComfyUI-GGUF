"""Block-streamed GGUF integer Tensor Core Linear kernels.

The raw GGUF tensor is borrowed through ComfyUI's CastBiasWeightContext.
Each CTA decodes only the current quant block, quantizes the matching
activation tile, runs integer WMMA, applies scale/offset correction, and
writes FP16/BF16 output. Expanded model weights are never cached.
"""

from __future__ import annotations

import ctypes as C
import hashlib
import logging
import os
from pathlib import Path
import tempfile
import threading

import torch

logger = logging.getLogger(__name__)

QTYPE_IDS = {"Q4_K": 0, "Q5_K": 1, "Q6_K": 2, "IQ3_S": 3, "IQ3_XXS": 4}
TYPE_SIZES = {"Q4_K": 144, "Q5_K": 176, "Q6_K": 210, "IQ3_S": 110, "IQ3_XXS": 98}
MIXED_BITS = {"Q4_K": 4, "IQ3_S": 4, "Q5_K": 8, "Q6_K": 8, "IQ3_XXS": 8}

_lock = threading.RLock()
_warned = set()
_dll_dirs = []
_loaded_modules = {}
_nvrtc_library = None
_driver_library = None


def _libraries():
    global _nvrtc_library, _driver_library
    with _lock:
        if _nvrtc_library is not None and _driver_library is not None:
            return _nvrtc_library, _driver_library
        torch_lib=Path(torch.__file__).resolve().parent/"lib"
        major=(torch.version.cuda or "13.0").split(".",1)[0]
        nvrtc_path=torch_lib/f"nvrtc64_{major}0_0.dll"
        if not nvrtc_path.is_file():
            raise RuntimeError(f"matching NVRTC DLL is missing: {nvrtc_path}")
        _dll_dirs.append(os.add_dll_directory(str(torch_lib)))
        nv=C.WinDLL(str(nvrtc_path));drv=C.WinDLL("nvcuda.dll")
        nv.nvrtcCreateProgram.argtypes=[C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p];nv.nvrtcCreateProgram.restype=C.c_int
        nv.nvrtcCompileProgram.argtypes=[C.c_void_p,C.c_int,C.POINTER(C.c_char_p)];nv.nvrtcCompileProgram.restype=C.c_int
        for fn in ("nvrtcGetProgramLogSize","nvrtcGetPTXSize","nvrtcGetCUBINSize"):
            getattr(nv,fn).argtypes=[C.c_void_p,C.POINTER(C.c_size_t)];getattr(nv,fn).restype=C.c_int
        for fn in ("nvrtcGetProgramLog","nvrtcGetPTX","nvrtcGetCUBIN"):
            getattr(nv,fn).argtypes=[C.c_void_p,C.c_void_p];getattr(nv,fn).restype=C.c_int
        nv.nvrtcDestroyProgram.argtypes=[C.POINTER(C.c_void_p)];nv.nvrtcDestroyProgram.restype=C.c_int
        nv.nvrtcVersion.argtypes=[C.POINTER(C.c_int),C.POINTER(C.c_int)];nv.nvrtcVersion.restype=C.c_int
        drv.cuCtxGetCurrent.argtypes=[C.POINTER(C.c_void_p)];drv.cuCtxGetCurrent.restype=C.c_int
        drv.cuModuleLoadData.argtypes=[C.POINTER(C.c_void_p),C.c_void_p];drv.cuModuleLoadData.restype=C.c_int
        drv.cuModuleGetFunction.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p];drv.cuModuleGetFunction.restype=C.c_int
        drv.cuLaunchKernel.argtypes=[C.c_void_p,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p];drv.cuLaunchKernel.restype=C.c_int
        _nvrtc_library,_driver_library=nv,drv
        return nv,drv


def warn_once(key, message):
    with _lock:
        if key in _warned:
            return
        _warned.add(key)
    logger.warning("ComfyUI-GGUF Tensor Core fallback (%s): %s", key, message)


def info_once(key, message):
    key = ("active",) + tuple(key)
    with _lock:
        if key in _warned:
            return
        _warned.add(key)
    logger.info("ComfyUI-GGUF Tensor Core Linear active: %s", message)


def _grid_values(qtype_name):
    import numpy as np
    from gguf import quants

    if qtype_name == "IQ3_S":
        quants.IQ3_S.init_grid()
        grid = np.asarray(quants.IQ3_S.grid).reshape(-1)
        name = "iq_grid"
        signs = None
    else:
        quants.IQ3_XXS.init_grid()
        grid = np.asarray(quants.IQ3_XXS.grid).reshape(-1)
        name = "iq_grid"
        signs = np.frombuffer(quants.IQ2_XXS.ksigns, dtype=np.uint8)
    declaration = (
        f"__device__ __constant__ unsigned char {name}[{len(grid)}] = "
        f"{{{','.join(str(int(value)) for value in grid)}}};"
    )
    if signs is None:
        sign_declaration = ""
    else:
        sign_declaration = (
            f"__device__ __constant__ unsigned char iq_signs[{len(signs)}] = "
            f"{{{','.join(str(int(value)) for value in signs)}}};"
        )
    return declaration, sign_declaration


def _grid_declarations():
    from gguf import quants
    import numpy as np

    quants.IQ3_S.init_grid()
    quants.IQ3_XXS.init_grid()
    grid_s = np.asarray(quants.IQ3_S.grid).reshape(-1)
    grid_x = np.asarray(quants.IQ3_XXS.grid).reshape(-1)
    signs = np.frombuffer(quants.IQ2_XXS.ksigns, dtype=np.uint8)
    return (
        f"__device__ __constant__ unsigned char iq3s_grid[{len(grid_s)}] = "
        f"{{{','.join(str(int(v)) for v in grid_s)}}};\n"
        f"__device__ __constant__ unsigned char iq3xxs_grid[{len(grid_x)}] = "
        f"{{{','.join(str(int(v)) for v in grid_x)}}};\n",
        f"__device__ __constant__ unsigned char iq3_signs[{len(signs)}] = "
        f"{{{','.join(str(int(v)) for v in signs)}}};",
    )


_KERNEL_SOURCE = r'''#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
using namespace nvcuda;
__GRID_DECL__
__SIGNS_DECL__

__device__ __forceinline__ float f16_at(const unsigned char* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}
__device__ __forceinline__ float load_scalar(const void* p, int i, int dtype) {
    if (dtype == 0) return __half2float(reinterpret_cast<const __half*>(p)[i]);
    if (dtype == 1) return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(p)[i]);
    return reinterpret_cast<const float*>(p)[i];
}
template<typename T> __device__ __forceinline__ T store_scalar(float v);
template<> __device__ __forceinline__ __half store_scalar<__half>(float v) { return __float2half_rn(v); }
template<> __device__ __forceinline__ __nv_bfloat16 store_scalar<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template<> __device__ __forceinline__ float store_scalar<float>(float v) { return v; }

// Return w = scale * code + offset in the original GGUF quantization.
__device__ __forceinline__ void decode(const unsigned char* p, int qt, int group, int i,
                                        int* code, float* scale, float* offset) {
    if (qt == 0 || qt == 1) { // Q4_K / Q5_K
        float d=f16_at(p), dm=f16_at(p+2); const unsigned char* s=p+4;
        unsigned sc, mn;
        if (group < 4) { sc=s[group]&63u; mn=s[4+group]&63u; }
        else { int j=group-4; sc=(s[8+j]&15u)|((s[j]>>2)&48u); mn=(s[8+j]>>4)|((s[4+j]>>2)&48u); }
        unsigned ql;
        if (qt == 0) { ql=(p[16+(group/2)*32+i]>>((group&1)*4))&15u; *code=int(ql)-8; }
        else { ql=(p[48+(group/2)*32+i]>>((group&1)*4))&15u; unsigned qh=(p[16+i]>>group)&1u; *code=int(ql|(qh<<4))-16; }
        *scale=d*float(sc); *offset=float(qt==0?8:16)*(*scale)-dm*float(mn); return;
    }
    if (qt == 2) { // Q6_K; scale changes every 16 values
        int ql_byte=(group/4)*64+(group&1)*32+i;
        int ql_shift=((group/2)&1)*4;
        unsigned ql=(p[ql_byte]>>ql_shift)&15u;
        unsigned qh=(p[128+(group/4)*32+i]>>((group&3)*2))&3u;
        *code=int(ql|(qh<<4))-32;
        int si=group*2+i/16;
        *scale=f16_at(p+208)*float(reinterpret_cast<const signed char*>(p+192)[si]);
        *offset=0.0f; return;
    }
    int pos=group*32+i, idx=pos/4, sub=pos&3;
    if (qt == 3) { // IQ3_S signed odd codebook -> exact affine INT4
        unsigned ix=p[2+idx]|(((p[66+idx/8]>>(idx&7))&1u)<<8);
        int mag=int(iq3s_grid[ix*4+sub]);
        int neg=(p[74+group*4+i/8]>>(i&7))&1;
        int signed_mag=neg?-mag:mag;
        *code=(signed_mag-1)/2;
        unsigned sp=p[106+group/2], nib=(group&1)?(sp>>4):(sp&15u);
        float base=f16_at(p)*float(1+2*nib);
        *scale=2.0f*base; *offset=base; return;
    }
    // IQ3_XXS has a nonuniform codebook; signed coefficients fit INT8.
    unsigned qs=p[2+idx];
    const unsigned char* sb=p+66+group*4;
    unsigned packed=unsigned(sb[0])|(unsigned(sb[1])<<8)|(unsigned(sb[2])<<16)|(unsigned(sb[3])<<24);
    float base=f16_at(p)*(0.5f+float((packed>>28)&15u))*0.5f;
    unsigned shifts[4]={0,7,14,21}, si=(packed>>shifts[i/8])&127u;
    int neg=(iq3_signs[si]>>(i&7))&1;
    int mag=int(iq3xxs_grid[qs*4+sub]);
    *code=neg?-mag:mag; *scale=base; *offset=0.0f;
}

template<int WB,typename IN,typename OUT>
__device__ void run_tile_multi(const unsigned char* raw,const IN* x,const IN* bias,OUT* y,
                               int M,int N,int K,int qt,int has_bias,int dtype,
                               int* a4,int* b4,signed char* a8,signed char* b8,
                               float* a_scale,float* a_sum,float* w_scale,float* w_offset,int* dots) {
    constexpr int MT=WB==4?8:16, NT=WB==4?8:16, TK=WB==4?32:16;
    constexpr int WM=2,WN=4,WARPS=WM*WN,OUTPUTS=MT*NT;
    const int warp=threadIdx.x/32,lane=threadIdx.x&31,wm=warp/WN,wn=warp%WN;
    const int mbase=blockIdx.y*MT*WM,nbase=blockIdx.x*NT*WN;
    const int block_bytes=qt==0?144:qt==1?176:qt==2?210:qt==3?110:98;
    float acc[8]={0,0,0,0,0,0,0,0};

    for(int k0=0;k0<K;k0+=TK) {
        if(wn==0) {
            const int row=WB==4?lane/4:lane/2,seg=WB==4?lane%4:lane%2;
            float vx[8],vmax=0.0f;int sum=0;
            #pragma unroll
            for(int j=0;j<8;j++) {
                int mi=mbase+wm*MT+row,ki=k0+seg*8+j;
                vx[j]=mi<M?load_scalar(x,mi*K+ki,dtype):0.0f;
                vmax=fmaxf(vmax,fabsf(vx[j]));
            }
            vmax=fmaxf(vmax,__shfl_xor_sync(0xffffffffu,vmax,1));
            if(WB==4)vmax=fmaxf(vmax,__shfl_xor_sync(0xffffffffu,vmax,2));
            float sx=vmax>0?vmax/(WB==4?7.0f:127.0f):1.0f;
            if(WB==4) {
                unsigned packed=0;
                #pragma unroll
                for(int j=0;j<8;j++){int v=max(-8,min(7,int(nearbyintf(vx[j]/sx))));packed|=(unsigned(v)&15u)<<(4*j);sum+=v;}
                a4[wm*32+row*4+seg]=int(packed);
            } else {
                #pragma unroll
                for(int j=0;j<8;j++){int v=max(-127,min(127,int(nearbyintf(vx[j]/sx))));a8[wm*256+row*16+seg*8+j]=(signed char)v;sum+=v;}
            }
            int total=sum+__shfl_xor_sync(0xffffffffu,sum,1);
            if(WB==4)total+=__shfl_xor_sync(0xffffffffu,total,2);
            if(seg==0){a_scale[wm*MT+row]=sx;a_sum[wm*MT+row]=float(total);}
        }

        const int col=lane%NT,bseg=lane/NT,oc=nbase+wn*NT+col;
        if(wm==0&&bseg==0) {
            if(oc<N) {
                int c=0;float sc=0,off=0;int group=(k0%256)/32;
                int i=WB==4?0:(k0%32);
                const unsigned char* block=raw+(oc*(K/256)+k0/256)*block_bytes;
                decode(block,qt,group,i,&c,&sc,&off);
                w_scale[wn*NT+col]=sc;w_offset[wn*NT+col]=off;
            } else {w_scale[wn*NT+col]=0;w_offset[wn*NT+col]=0;}
        }
        if(wm==0&&WB==4) {
            unsigned packed=0;
            if(oc<N) {
                int group=(k0%256)/32;const unsigned char* block=raw+(oc*(K/256)+k0/256)*block_bytes;
                #pragma unroll
                for(int j=0;j<8;j++){int c=0;float sc,off;decode(block,qt,group,bseg*8+j,&c,&sc,&off);packed|=(unsigned(c)&15u)<<(4*j);}
            }
            b4[wn*32+col*4+bseg]=int(packed);
        } else if(wm==0) {
            if(oc<N) {
                int group=(k0%256)/32;const unsigned char* block=raw+(oc*(K/256)+k0/256)*block_bytes;
                #pragma unroll
                for(int j=0;j<8;j++){int c=0;float sc,off;decode(block,qt,group,(k0%32)+bseg*8+j,&c,&sc,&off);b8[wn*256+col*16+bseg*8+j]=(signed char)c;}
            } else {
                #pragma unroll
                for(int j=0;j<8;j++)b8[wn*256+col*16+bseg*8+j]=0;
            }
        }
        __syncthreads();
        if(WB==4) {
            wmma::fragment<wmma::matrix_a,8,8,32,wmma::experimental::precision::s4,wmma::row_major>A;
            wmma::fragment<wmma::matrix_b,8,8,32,wmma::experimental::precision::s4,wmma::col_major>B;
            wmma::fragment<wmma::accumulator,8,8,32,int>C;wmma::fill_fragment(C,0);
            wmma::load_matrix_sync(A,reinterpret_cast<const wmma::experimental::precision::s4*>(a4+wm*32),32);
            wmma::load_matrix_sync(B,reinterpret_cast<const wmma::experimental::precision::s4*>(b4+wn*32),32);
            wmma::mma_sync(C,A,B,C);wmma::store_matrix_sync(dots+warp*64,C,8,wmma::mem_row_major);
        } else {
            wmma::fragment<wmma::matrix_a,16,16,16,signed char,wmma::row_major>A;
            wmma::fragment<wmma::matrix_b,16,16,16,signed char,wmma::col_major>B;
            wmma::fragment<wmma::accumulator,16,16,16,int>C;wmma::fill_fragment(C,0);
            wmma::load_matrix_sync(A,a8+wm*256,16);wmma::load_matrix_sync(B,b8+wn*256,16);
            wmma::mma_sync(C,A,B,C);wmma::store_matrix_sync(dots+warp*256,C,16,wmma::mem_row_major);
        }
        __syncwarp();
        const int count=WB==4?2:8;
        #pragma unroll
        for(int z=0;z<count;z++) {
            int idx=lane+z*32,row=idx/NT,c=idx%NT;
            float v=(float(dots[warp*OUTPUTS+idx])*w_scale[wn*NT+c]+
                     a_sum[wm*MT+row]*w_offset[wn*NT+c])*a_scale[wm*MT+row];
            acc[z]+=v;
        }
        __syncthreads();
    }
    const int count=WB==4?2:8;
    #pragma unroll
    for(int z=0;z<count;z++) {
        int idx=lane+z*32,row=idx/NT,c=idx%NT;
        int mi=mbase+wm*MT+row,ni=nbase+wn*NT+c;
        if(mi<M&&ni<N){float v=acc[z];if(has_bias)v+=load_scalar(bias,ni,dtype);y[mi*N+ni]=store_scalar<OUT>(v);}
    }
}

#define DEFINE_KERNEL(NAME,WB,IN,OUT) \
extern "C" __global__ void NAME(const unsigned char*r,const IN*x,const IN*b,OUT*y,int M,int N,int K,int q,int hb,int dt){ \
    __shared__ __align__(16) int a4[64],b4[128]; \
    __shared__ __align__(16) signed char a8[512],b8[1024]; \
    __shared__ float as[32],asum[32],ws[64],wo[64]; \
    __shared__ __align__(16) int dots[2048]; \
    run_tile_multi<WB>(r,x,b,y,M,N,K,q,hb,dt,a4,b4,a8,b8,as,asum,ws,wo,dots); \
}
DEFINE_KERNEL(gguf_i4_f16,4,__half,__half)
DEFINE_KERNEL(gguf_i4_bf16,4,__nv_bfloat16,__nv_bfloat16)
DEFINE_KERNEL(gguf_i4_f32,4,float,float)
DEFINE_KERNEL(gguf_i8_f16,8,__half,__half)
DEFINE_KERNEL(gguf_i8_bf16,8,__nv_bfloat16,__nv_bfloat16)
DEFINE_KERNEL(gguf_i8_f32,8,float,float)
#undef DEFINE_KERNEL
'''


def _generated_source():
    grid_s, _ = _grid_values("IQ3_S")
    grid_x, signs = _grid_values("IQ3_XXS")
    return (_KERNEL_SOURCE.replace("__GRID_DECL__", grid_s.replace("iq_grid", "iq3s_grid")+"\n"+
                                  grid_x.replace("iq_grid", "iq3xxs_grid"))
            .replace("__SIGNS_DECL__", signs.replace("iq_signs", "iq3_signs")))


def _check(status, what):
    if status:
        raise RuntimeError(f"{what} failed with CUDA/NVRTC status {status}")


def _compile_and_load(device):
    if os.name != "nt": raise RuntimeError("NVRTC bridge currently supports Windows only")
    nv,drv=_libraries()
    torch_lib=Path(torch.__file__).resolve().parent/"lib"
    src=_generated_source();arch=torch.cuda.get_device_capability(device);cc=arch[0]*10+arch[1]
    nv_major=C.c_int();nv_minor=C.c_int();_check(nv.nvrtcVersion(C.byref(nv_major),C.byref(nv_minor)),"nvrtcVersion")
    candidates=[];cuda_home=os.environ.get("CUDA_PATH")
    if cuda_home:candidates.append(Path(cuda_home)/"include")
    toolkit=Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if toolkit.is_dir():candidates += [p/"include" for p in sorted(toolkit.glob("v*"),reverse=True)]
    include=next((p for p in candidates if (p/"mma.h").is_file()),None)
    if include is None:raise RuntimeError("CUDA mma.h header was not found")
    compile_id=f"{cc}|{torch.version.cuda}|nvrtc-{nv_major.value}.{nv_minor.value}|{include}"
    digest=hashlib.sha256(src.encode()+compile_id.encode()).hexdigest()
    cache=Path(os.environ.get("LOCALAPPDATA",tempfile.gettempdir()))/"ComfyUI-Shared"/"cache"/"comfyui-gguf-tensor-core"
    cache.mkdir(parents=True,exist_ok=True);cubin_path=cache/f"{digest}.cubin";ptx_path=cache/f"{digest}.ptx"
    if cubin_path.is_file():code=cubin_path.read_bytes();code_kind="cubin"
    elif ptx_path.is_file():code=ptx_path.read_bytes();code_kind="ptx"
    else:
        prog=C.c_void_p();_check(nv.nvrtcCreateProgram(C.byref(prog),src.encode(),b"gguf_tensor_core.cu",0,None,None),"nvrtcCreateProgram")
        try:
            opts=[f"--gpu-architecture=sm_{cc}".encode(),b"--std=c++17",f"--include-path={include}".encode()]
            status=nv.nvrtcCompileProgram(prog,len(opts),(C.c_char_p*len(opts))(*opts));sz=C.c_size_t()
            _check(nv.nvrtcGetProgramLogSize(prog,C.byref(sz)),"nvrtcGetProgramLogSize");log=C.create_string_buffer(max(sz.value,1))
            _check(nv.nvrtcGetProgramLog(prog,log),"nvrtcGetProgramLog")
            if status:raise RuntimeError("NVRTC compile failed: "+log.value.decode("utf-8","replace"))
            _check(nv.nvrtcGetCUBINSize(prog,C.byref(sz)),"nvrtcGetCUBINSize")
            if sz.value:
                buf=C.create_string_buffer(sz.value);_check(nv.nvrtcGetCUBIN(prog,buf),"nvrtcGetCUBIN")
                code=buf.raw[:sz.value];code_kind="cubin";out_path=cubin_path
            else:
                _check(nv.nvrtcGetPTXSize(prog,C.byref(sz)),"nvrtcGetPTXSize")
                buf=C.create_string_buffer(sz.value);_check(nv.nvrtcGetPTX(prog,buf),"nvrtcGetPTX")
                code=buf.value;code_kind="ptx";out_path=ptx_path
            tmp=out_path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_bytes(code);tmp.replace(out_path)
        finally:nv.nvrtcDestroyProgram(C.byref(prog))
    ctx=C.c_void_p();_check(drv.cuCtxGetCurrent(C.byref(ctx)),"cuCtxGetCurrent")
    if not ctx.value:raise RuntimeError("PyTorch CUDA context is not current")
    module=C.c_void_p();codebuf=C.create_string_buffer(code+b"\0");_check(drv.cuModuleLoadData(C.byref(module),C.cast(codebuf,C.c_void_p)),"cuModuleLoadData")
    funcs={}
    for name in ("gguf_i4_f16","gguf_i4_bf16","gguf_i4_f32","gguf_i8_f16","gguf_i8_bf16","gguf_i8_f32"):
        fn=C.c_void_p();_check(drv.cuModuleGetFunction(C.byref(fn),module,name.encode()),"cuModuleGetFunction");funcs[name]=fn
    return {"driver":drv,"module":module,"codebuf":codebuf,"functions":funcs,"context":ctx.value,"device":device,"cc":cc,"code":code,"code_kind":code_kind}


def compile_all_devices():
    """NVRTC/PTX compile and module-load check; does not launch a kernel."""
    results=[]
    for device in range(torch.cuda.device_count()):
        major,minor=torch.cuda.get_device_capability(device)
        if (major,minor)<(7,5):results.append({"device":device,"supported":False,"reason":"INT4 WMMA requires SM 7.5+"});continue
        torch.cuda.current_stream(device)
        module=_compile_and_load(device)
        results.append({"device":device,"name":torch.cuda.get_device_name(device),"compute_capability":module["cc"],
                        "code_kind":module["code_kind"],"code_bytes":len(module["code"]),
                        "kernels":sorted(module["functions"])})
    return results


def _get_module(device):
    torch.cuda.current_stream(device);drv=_libraries()[1]
    ctx=C.c_void_p();_check(drv.cuCtxGetCurrent(C.byref(ctx)),"cuCtxGetCurrent");key=(device,ctx.value)
    with _lock:
        if key not in _loaded_modules:_loaded_modules[key]=_compile_and_load(device)
        return _loaded_modules[key]


def launch_linear(raw,input_2d,bias,output_2d,qtype_name,bits):
    if qtype_name not in QTYPE_IDS or bits not in (4,8):raise ValueError("unsupported GGUF Tensor Core configuration")
    if input_2d.dtype not in (torch.float16,torch.bfloat16,torch.float32):raise ValueError("input must be FP16/BF16/FP32")
    n,k=output_2d.shape[1],input_2d.shape[1]
    if k%256:raise ValueError("K dimension must align to the 256-value GGUF block")
    if not raw.is_cuda or raw.device!=input_2d.device:raise ValueError("packed weight and input devices differ")
    capability=torch.cuda.get_device_capability(input_2d.device)
    if capability < (7,5):raise ValueError(f"INT4 WMMA requires SM 7.5+, got {capability}")
    if raw.dtype!=torch.uint8:raw=raw.view(torch.uint8)
    if not raw.is_contiguous():raise ValueError("packed GGUF data must be contiguous")
    expected=n*(k//256)*TYPE_SIZES[qtype_name]
    if raw.numel()<expected:raise ValueError(f"packed storage too short: {raw.numel()} < {expected}")
    if bias is not None and (not bias.is_cuda or bias.device!=input_2d.device or bias.dtype!=input_2d.dtype):
        raise ValueError("bias must match the input dtype/device")
    device=input_2d.device.index or 0;mod=_get_module(device);drv=mod["driver"]
    dtype="f16" if input_2d.dtype==torch.float16 else "bf16" if input_2d.dtype==torch.bfloat16 else "f32"
    fn=mod["functions"][f"gguf_i{bits}_{dtype}"]
    m=input_2d.shape[0];dt=0 if input_2d.dtype==torch.float16 else 1 if input_2d.dtype==torch.bfloat16 else 2
    vals=[C.c_void_p(raw.data_ptr()),C.c_void_p(input_2d.data_ptr()),C.c_void_p(0 if bias is None else bias.data_ptr()),
          C.c_void_p(output_2d.data_ptr()),C.c_int(m),C.c_int(n),C.c_int(k),C.c_int(QTYPE_IDS[qtype_name]),C.c_int(bias is not None),C.c_int(dt)]
    params=(C.c_void_p*len(vals))(*[C.cast(C.pointer(v),C.c_void_p) for v in vals])
    tile_m=(8 if bits==4 else 16)*2;tile_n=(8 if bits==4 else 16)*4
    gx=(n+tile_n-1)//tile_n;gy=(m+tile_m-1)//tile_m
    stream=torch.cuda.current_stream(input_2d.device).cuda_stream
    _check(drv.cuLaunchKernel(fn,gx,gy,1,256,1,1,0,C.c_void_p(stream),params,None),"cuLaunchKernel")
