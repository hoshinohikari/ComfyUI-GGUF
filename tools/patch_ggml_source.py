"""Apply pinned ComfyUI borrowed-pointer and typed-activation patches to ggml."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


EXPECTED_COMMIT = "d9fe7bff3d2bf4d077e25c654486d02c321734aa"
BRIDGE_INCLUDE = '#include "comfyui_gguf_mmq_external.inc"'


def patch_mmq_activations(source: Path, check_only: bool) -> None:
    quantize_header = source / "src" / "ggml-cuda" / "quantize.cuh"
    header = quantize_header.read_text(encoding="utf-8")
    typed_decl = "void quantize_mmq_q8_1_cuda_typed("
    if typed_decl not in header:
        old_decl = (
            "void quantize_mmq_q8_1_cuda(\n"
            "        const float * x, const int32_t * ids, void * vy,\n"
            "        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,\n"
            "        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);\n"
        )
        if header.count(old_decl) != 1:
            raise ValueError("Pinned quantize.cuh declaration differs from the reviewed activation patch")
        typed = (
            old_decl +
            "\nvoid quantize_mmq_q8_1_cuda_typed(\n"
            "        const void * x, ggml_type type_src1, void * vy, ggml_type type_src0,\n"
            "        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,\n"
            "        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);\n"
        )
        header = header.replace(old_decl, typed, 1)
        if not check_only:
            quantize_header.write_text(header, encoding="utf-8", newline="\n")

    quantize_path = source / "src" / "ggml-cuda" / "quantize.cu"
    text = quantize_path.read_text(encoding="utf-8")
    if "load_mmq_q8_1_x4" not in text:
        include = '#include "quantize.cuh"\n'
        if text.count(include) != 1:
            raise ValueError("Pinned quantize.cu include context differs from the reviewed activation patch")
        text = text.replace(include, include + '#include "convert.cuh"\n', 1)
        kernel_decl = (
            "template <mmq_q8_1_ds_layout ds_layout, bool scatter>\n"
            "static __global__ void quantize_mmq_q8_1(\n"
            "        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,"
        )
        helper = (
            "template <typename src_t>\n"
            "static __device__ __forceinline__ float4 load_mmq_q8_1_x4(const src_t * x, const int64_t index) {\n"
            "    return make_float4(ggml_cuda_cast<float>(x[index + 0]), ggml_cuda_cast<float>(x[index + 1]),\n"
            "                       ggml_cuda_cast<float>(x[index + 2]), ggml_cuda_cast<float>(x[index + 3]));\n"
            "}\n\n"
            "template <>\n"
            "static __device__ __forceinline__ float4 load_mmq_q8_1_x4<float>(const float * x, const int64_t index) {\n"
            "    return reinterpret_cast<const float4 *>(x + index)[0];\n"
            "}\n\n"
        )
        if text.count(kernel_decl) != 1:
            raise ValueError("Pinned Q8_1 kernel signature differs from the reviewed activation patch")
        text = text.replace("// scatter: grid over tokens, quantize once, write to all the token's compact rows\n", helper + "// scatter: grid over tokens, quantize once, write to all the token's compact rows\n", 1)
        text = text.replace(
            kernel_decl,
            "template <mmq_q8_1_ds_layout ds_layout, bool scatter, typename src_t>\n"
            "static __global__ void quantize_mmq_q8_1(\n"
            "        const src_t * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,",
            1,
        )
        old_load = (
            "    const float4 * x4 = (const float4 *) x;\n"
            "    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;"
        )
        new_load = "    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;"
        if text.count(old_load) != 1:
            raise ValueError("Pinned Q8_1 source load declaration differs from the reviewed activation patch")
        text = text.replace(old_load, new_load, 1)
        old_xi = "const float4 xi = i0 < ne00 ? x4[(base_idx + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);"
        new_xi = "const float4 xi = i0 < ne00 ? load_mmq_q8_1_x4(x + base_idx, i00) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);"
        if text.count(old_xi) != 1:
            raise ValueError("Pinned Q8_1 vector load differs from the reviewed activation patch")
        text = text.replace(old_xi, new_xi, 1)

        start = text.index("void quantize_mmq_q8_1_cuda(\n")
        end_marker = "// Each thread quantizes 4 values of one column, see quantize.cuh for the layout."
        end = text.index(end_marker, start)
        launcher = r'''template <typename src_t>
static void quantize_mmq_q8_1_cuda_impl(
        const src_t * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % QK8_1_MMQ == 0);
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, false, src_t><<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, 0);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, false, src_t><<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, 0);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, false, src_t><<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, 0);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    quantize_mmq_q8_1_cuda_impl<float>(x, ids, vy, type_src0, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, stream);
}

void quantize_mmq_q8_1_cuda_typed(
        const void * x, const ggml_type type_src1, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    switch (type_src1) {
        case GGML_TYPE_F32:
            quantize_mmq_q8_1_cuda_impl<float>(static_cast<const float *>(x), nullptr, vy, type_src0, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, stream);
            break;
        case GGML_TYPE_F16:
            quantize_mmq_q8_1_cuda_impl<half>(static_cast<const half *>(x), nullptr, vy, type_src0, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, stream);
            break;
        case GGML_TYPE_BF16:
            quantize_mmq_q8_1_cuda_impl<nv_bfloat16>(static_cast<const nv_bfloat16 *>(x), nullptr, vy, type_src0, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, stream);
            break;
        default:
            GGML_ABORT("unsupported MMQ activation type");
    }
}

'''
        text = text[:start] + launcher + text[end:]
        text = text.replace("quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, true>", "quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, true, float>")
        text = text.replace("quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, true>", "quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, true, float>")
        if not check_only:
            quantize_path.write_text(text, encoding="utf-8", newline="\n")

    # In check mode, the signatures and helper must all be present.
    if typed_decl not in header or "load_mmq_q8_1_x4" not in text or "GGML_TYPE_BF16" not in text:
        raise ValueError("GGML typed-activation patch is incomplete")

    mmq_path = source / "src" / "ggml-cuda" / "mmq.cu"
    mmq = mmq_path.read_text(encoding="utf-8")
    required = "quantize_mmq_q8_1_cuda_typed(src1_data, src1->type"
    if required not in mmq:
        old_assert = "GGML_ASSERT(        src1->type == GGML_TYPE_F32);"
        new_assert = "GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);"
        if mmq.count(old_assert) != 1:
            raise ValueError("Pinned MMQ activation type assertion differs from the reviewed patch")
        mmq = mmq.replace(old_assert, new_assert, 1)
        old_src = "    const float * src1_d = (const float *) src1->data;"
        new_src = (
            "    const void * src1_data = src1->data;\n"
            "    const float * src1_d = src1->type == GGML_TYPE_F32 ? (const float *) src1->data : nullptr;\n"
            "    GGML_ASSERT(!ids || src1->type == GGML_TYPE_F32);"
        )
        if mmq.count(old_src) != 1:
            raise ValueError("Pinned MMQ activation data pointer differs from the reviewed patch")
        mmq = mmq.replace(old_src, new_src, 1)
        old_fp4 = "    const bool use_native_fp4 = prec_src1 == GGML_PREC_Q4;"
        if mmq.count(old_fp4) != 1:
            raise ValueError("Pinned MMQ precision selection differs from the reviewed patch")
        mmq = mmq.replace(old_fp4, old_fp4 + "\n    GGML_ASSERT(!use_native_fp4 || src1->type == GGML_TYPE_F32);", 1)
        old_q8 = "quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,"
        new_q8 = "quantize_mmq_q8_1_cuda_typed(src1_data, src1->type, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,"
        if mmq.count(old_q8) != 1:
            raise ValueError("Pinned MMQ Q8_1 launch differs from the reviewed patch")
        mmq = mmq.replace(old_q8, new_q8, 1)
        if not check_only:
            mmq_path.write_text(mmq, encoding="utf-8", newline="\n")
    if required not in mmq or "GGML_TYPE_BF16" not in mmq:
        raise ValueError("GGML MMQ activation type patch is incomplete")


def patch_mmq_bf16_output(source: Path, check_only: bool) -> None:
    mmq_header = source / "src" / "ggml-cuda" / "mmq.cuh"
    text = mmq_header.read_text(encoding="utf-8")
    marker = "ggml_cuda_mmq_store_output"
    if marker not in text:
        typedef = (
            "typedef void (*ggml_cuda_mmq_write_back_t)(const float * __restrict__ sum, const int32_t * __restrict__ ids_dst,\n"
            "    float * __restrict__ dst, const float * __restrict__ y_scale, const int stride, const int i_max, const int j_sup,\n"
            "    const ggml_glu_op glu_op, const float glu_limit, const float * __restrict__ x_scale, const float * __restrict__ gate_scale);"
        )
        helpers = typedef + """

static __device__ __forceinline__ float * ggml_cuda_mmq_offset_output(
        float * dst, const int64_t offset, const ggml_type dst_type) {
    if (dst_type == GGML_TYPE_BF16) {
        return reinterpret_cast<float *>(reinterpret_cast<nv_bfloat16 *>(dst) + offset);
    }
    return dst + offset;
}

template <ggml_type type>
static __device__ __forceinline__ void ggml_cuda_mmq_store_output(
        float * dst, const int64_t index, const float value, const ggml_type dst_type) {
    if constexpr (type == GGML_TYPE_Q8_0) {
        if (dst_type == GGML_TYPE_BF16) {
            reinterpret_cast<nv_bfloat16 *>(dst)[index] = __float2bfloat16_rn(value);
        } else {
            dst[index] = value;
        }
    } else {
        dst[index] = value;
    }
}
"""
        if text.count(typedef) != 1:
            raise ValueError("Pinned MMQ writeback function pointer differs from the reviewed BF16 patch")
        text = text.replace(typedef, helpers, 1)

        old_param_patterns = (
            "float * __restrict__ dst, const float * __restrict__ y_scale",
            "const float * __restrict__ sum, const int32_t * __restrict__ ids_dst, float * __restrict__ dst,\n"
            "        const float * __restrict__ y_scale",
            "const float * __restrict__ sum, const int * __restrict__ ids_dst, float * __restrict__ dst,\n"
            "            const float * __restrict__ y_scale",
        )
        for index, old_param in enumerate(old_param_patterns):
            if text.count(old_param) != 1:
                raise ValueError("Pinned MMQ writeback signatures differ from the reviewed BF16 patch")
            if index == 0:
                new_param = "float * __restrict__ dst, const ggml_type dst_type, const float * __restrict__ y_scale"
            else:
                new_param = old_param.replace(
                    "float * __restrict__ dst,\n",
                    "float * __restrict__ dst, const ggml_type dst_type,\n",
                    1,
                )
            text = text.replace(old_param, new_param, 1)

        stores = {
            "dst[ids_dst[j]*stride + i] = value * scale;":
                "if constexpr (type == GGML_TYPE_Q8_0) {\n"
                "    ggml_cuda_mmq_store_output<type>(dst, ids_dst[j]*stride + i, value * scale, dst_type);\n"
                "} else {\n"
                "    dst[ids_dst[j]*stride + i] = value * scale;\n"
                "}",
            "dst[ids_dst[j]*stride + i] = ggml_cuda_op_glu_single(glu_op, gate, up, glu_limit);":
                "if constexpr (type == GGML_TYPE_Q8_0) {\n"
                "    ggml_cuda_mmq_store_output<type>(dst, ids_dst[j]*stride + i,\n"
                "        ggml_cuda_op_glu_single(glu_op, gate, up, glu_limit), dst_type);\n"
                "} else {\n"
                "    dst[ids_dst[j]*stride + i] = ggml_cuda_op_glu_single(glu_op, gate, up, glu_limit);\n"
                "}",
            "dst[ids_dst[j]*stride + i] = value * scale_up;":
                "if constexpr (type == GGML_TYPE_Q8_0) {\n"
                "    ggml_cuda_mmq_store_output<type>(dst, ids_dst[j]*stride + i, value * scale_up, dst_type);\n"
                "} else {\n"
                "    dst[ids_dst[j]*stride + i] = value * scale_up;\n"
                "}",
        }
        for old, new in stores.items():
            if text.count(old) != 1:
                raise ValueError(f"Pinned MMQ store differs from the reviewed BF16 patch: {old}")
            text = text.replace(old, new, 1)

        process_params = (
            "const int * __restrict__ ids_dst, float * __restrict__ dst, float * __restrict__ tmp_fixup,"
        )
        if text.count(process_params) != 1:
            raise ValueError("Pinned MMQ process-tile signature differs from the reviewed BF16 patch")
        text = text.replace(
            process_params,
            "const int * __restrict__ ids_dst, float * __restrict__ dst, const ggml_type dst_type, float * __restrict__ tmp_fixup,",
            1,
        )
        kernel_params = (
            "const ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec, float * __restrict__ dst, float * __restrict__ tmp_fixup,"
        )
        if text.count(kernel_params) != 1:
            raise ValueError("Pinned MMQ kernel signature differs from the reviewed BF16 patch")
        text = text.replace(
            kernel_params,
            "const ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec, float * __restrict__ dst, const ggml_type dst_type,\n"
            "        const bool force_tiling, float * __restrict__ tmp_fixup,",
            1,
        )
        process_tile = (
            "ids_dst_shared, dst + offset_dst, tmp_fixup, y_scale_tile,"
        )
        if text.count(process_tile) != 3:
            raise ValueError("Pinned MMQ process-tile calls differ from the reviewed BF16 patch")
        text = text.replace(
            process_tile,
            "ids_dst_shared, ggml_cuda_mmq_offset_output(dst, offset_dst, dst_type), dst_type, tmp_fixup, y_scale_tile,",
            2,
        )
        text = text.replace(
            process_tile,
            "ids_dst_shared, dst + offset_dst, dst_type, tmp_fixup, y_scale_tile,",
            1,
        )
        fixup_call = (
            "write_back(sum, ids_dst, tmp_fixup + blockIdx.x*(J*I), y_scale, I, I, J, glu_op, glu_limit, x_scale, gate_scale);"
        )
        direct_call = (
            "write_back(sum, ids_dst, dst, y_scale, stride_col_dst, tile_x_i_max, tile_y_j_sup, glu_op, glu_limit, x_scale, gate_scale);"
        )
        if text.count(fixup_call) != 1 or text.count(direct_call) != 1:
            raise ValueError("Pinned MMQ writeback calls differ from the reviewed BF16 patch")
        text = text.replace(
            fixup_call,
            "write_back(sum, ids_dst, tmp_fixup + blockIdx.x*(J*I), GGML_TYPE_F32, y_scale, I, I, J, glu_op, glu_limit, x_scale, gate_scale);",
            1,
        )
        text = text.replace(
            direct_call,
            "write_back(sum, ids_dst, dst, dst_type, y_scale, stride_col_dst, tile_x_i_max, tile_y_j_sup, glu_op, glu_limit, x_scale, gate_scale);",
            1,
        )
        text = text.replace(
            "if constexpr (has_fusion || !ggml_cuda_mmq_get_stream_k(type, J, fallback, prec_src1)) {",
            "if (has_fusion || !ggml_cuda_mmq_get_stream_k(type, J, fallback, prec_src1) || force_tiling) {",
            1,
        )
        args_field = "const ggml_cuda_mm_id_tile_spec * tile_spec; float * dst; const float * y_scale;"
        if text.count(args_field) != 1:
            raise ValueError("Pinned MMQ launch arguments differ from the reviewed BF16 patch")
        text = text.replace(
            args_field,
            "const ggml_cuda_mm_id_tile_spec * tile_spec; float * dst; ggml_type dst_type; const float * y_scale;",
            1,
        )

        launch_tiled = (
            "(args.x, (const char *) args.fusion.gate, args.y, args.ids_dst, args.tile_spec, args.dst, nullptr,"
        )
        stream_launch = (
            "(args.x, (const char *) args.fusion.gate, args.y, args.ids_dst, args.tile_spec, args.dst, tmp_fixup.ptr,"
        )
        if text.count(launch_tiled) != 1 or text.count(stream_launch) != 1:
            raise ValueError("Pinned MMQ kernel launches differ from the reviewed BF16 patch")
        text = text.replace(
            launch_tiled,
            "(args.x, (const char *) args.fusion.gate, args.y, args.ids_dst, args.tile_spec, args.dst, args.dst_type, false, nullptr,",
            1,
        )
        text = text.replace(
            stream_launch,
            "(args.x, (const char *) args.fusion.gate, args.y, args.ids_dst, args.tile_spec, args.dst, args.dst_type, false, tmp_fixup.ptr,",
            1,
        )
        stream_setup = (
            "    ggml_cuda_pool & pool = ctx.pool(id);\n"
            "    ggml_cuda_pool_alloc<float> tmp_fixup(pool);"
        )
        direct_tiled = """    // Stream-K fixup accumulates partial tiles into the destination. Keep its
    // FP32 accumulation semantics by using a full-K tiled launch for BF16 output.
    if (fixup_needed && args.dst_type == GGML_TYPE_BF16) {
        mul_mat_q<type, J, fallback, prec_src1, has_fusion><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
            (args.x, (const char *) args.fusion.gate, args.y, args.ids_dst, args.tile_spec, args.dst, args.dst_type, true, nullptr,
             args.y_scale, args.fusion.glu_op, args.fusion.glu_limit, (const float *) args.fusion.x_scale, (const float *) args.fusion.gate_scale,
             blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
             channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
             sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
             ntj_fd);
        return;
    }

"""
        if text.count(stream_setup) != 1:
            raise ValueError("Pinned MMQ Stream-K setup differs from the reviewed BF16 patch")
        text = text.replace(stream_setup, direct_tiled + stream_setup, 1)

    old_writeback_typedef = (
        "typedef void (*ggml_cuda_mmq_write_back_t)(const float * __restrict__ sum, const int32_t * __restrict__ ids_dst,\n"
        "    float * __restrict__ dst, const float * __restrict__ y_scale"
    )
    new_writeback_typedef = old_writeback_typedef.replace(
        "float * __restrict__ dst, const float *",
        "float * __restrict__ dst, const ggml_type dst_type, const float *",
        1,
    )
    if old_writeback_typedef in text:
        if check_only:
            raise ValueError("GGML BF16 MMQ writeback typedef patch is incomplete")
        text = text.replace(old_writeback_typedef, new_writeback_typedef, 1)
    if not check_only:
        mmq_header.write_text(text, encoding="utf-8", newline="\n")

    required_header = (
        "ggml_cuda_mmq_store_output",
        "ggml_cuda_mmq_offset_output",
        new_writeback_typedef,
        "const bool force_tiling",
        "args.dst_type == GGML_TYPE_BF16",
    )
    if any(marker not in text for marker in required_header):
        raise ValueError("GGML BF16 MMQ writeback patch is incomplete")

    mmq_source = source / "src" / "ggml-cuda" / "mmq.cu"
    source_text = mmq_source.read_text(encoding="utf-8")
    new_assert = (
        "GGML_ASSERT(dst->type == GGML_TYPE_F32 ||\n"
        "        (dst->type == GGML_TYPE_BF16 && src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_BF16));"
    )
    if new_assert not in source_text:
        old_assert = "GGML_ASSERT(        dst->type  == GGML_TYPE_F32);"
        if source_text.count(old_assert) != 1:
            raise ValueError("Pinned MMQ output assertion differs from the reviewed BF16 patch")
        source_text = source_text.replace(old_assert, new_assert, 1)
        initializers = (
            "src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,",
            "src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), tile_spec.get(), dst_d,",
        )
        for old in initializers:
            if source_text.count(old) != 1:
                raise ValueError(f"Pinned MMQ argument initializer differs from the reviewed BF16 patch: {old}")
            source_text = source_text.replace(old, old[:-1] + " dst->type,", 1)
        if not check_only:
            mmq_source.write_text(source_text, encoding="utf-8", newline="\n")
    if new_assert not in source_text or source_text.count("dst->type,") < 2:
        raise ValueError("GGML BF16 MMQ output patch is incomplete")


def apply_patch(source: Path, bridge_include: Path, check_only: bool) -> None:
    source = source.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise ValueError(f"Expected ggml {EXPECTED_COMMIT}, found {commit}")

    mmq_path = source / "src" / "ggml-cuda" / "mmq.cu"
    text = mmq_path.read_text(encoding="utf-8")
    if BRIDGE_INCLUDE in text:
        if "if (t && t->buffer && ggml_backend_buffer_get_usage(t->buffer)" not in text:
            raise ValueError("Bridge include exists but borrowed-buffer guard is missing")
        if "#include <map>" not in text:
            if check_only:
                raise ValueError("Bridge include exists but the external-stream map header is missing")
            text = text.replace("#include <unordered_map>\n", "#include <unordered_map>\n#include <map>\n", 1)
            mmq_path.write_text(text, encoding="utf-8", newline="\n")
        if not (mmq_path.parent / bridge_include.name).exists() and not check_only:
            raise ValueError("Bridge include is already applied but its .inc file is missing")
        if not check_only:
            (mmq_path.parent / bridge_include.name).write_bytes(bridge_include.read_bytes())
    else:
        include_block = "#include <cstdint>\n"
        bridge_includes = (
            "#include <cstdint>\n"
            "#include <mutex>\n"
            "#include <unordered_map>\n"
            "#include <map>\n"
            "#include <memory>\n"
            "#include <string>\n"
            "#include <stdexcept>\n"
            "#include <algorithm>\n"
            "#include <cstdio>\n"
            "#include <cstring>\n"
        )
        buffer_check = "if (t && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {"
        borrowed_buffer_check = "if (t && t->buffer && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {"
        if text.count(include_block) != 1 or text.count(buffer_check) != 1:
            raise ValueError("Pinned ggml source context differs from the reviewed bridge patch")
        text = text.replace(include_block, bridge_includes, 1)
        text = text.replace(buffer_check, borrowed_buffer_check, 1)
        text = text.rstrip() + "\n\n" + BRIDGE_INCLUDE + "\n"
        if not check_only:
            mmq_path.write_text(text, encoding="utf-8", newline="\n")
            (mmq_path.parent / bridge_include.name).write_bytes(bridge_include.read_bytes())

    patch_mmq_activations(source, check_only)
    patch_mmq_bf16_output(source, check_only)
    print("ggml MMQ bridge, typed activations, and Q8_0 BF16 output path can be applied" if check_only else "Applied ggml MMQ bridge, typed activations, and Q8_0 BF16 output path")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--bridge-include",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "native" / "ggml" / "comfyui_gguf_mmq_external.inc",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    apply_patch(args.source, args.bridge_include, args.check)


if __name__ == "__main__":
    main()
