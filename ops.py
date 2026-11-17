# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import contextlib
import gguf
import torch
import logging
from dataclasses import dataclass

import comfy.ops
import comfy.lora
import comfy.model_management
from comfy.quant_ops import QuantizedTensor
from comfy_kitchen.tensor import QuantizedLayout, register_layout_class
from .dequant import GGUF_QUANTIZED_LAYOUT, dequantize, dequantize_tensor, is_quantized

def chained_hasattr(obj, chained_attr):
    probe = obj
    for attr in chained_attr.split('.'):
        if hasattr(probe, attr):
            probe = getattr(probe, attr)
        else:
            return False
    return True

# A bakcward and forward compatible way to get `torch.compiler.disable`.
def get_torch_compiler_disable_decorator():
    def dummy_decorator(*args, **kwargs):
        def noop(x):
            return x
        return noop

    from packaging import version

    if not chained_hasattr(torch, "compiler.disable"):
        logging.info("ComfyUI-GGUF: Torch too old for torch.compile - bypassing")
        return dummy_decorator # torch too old
    elif version.parse(torch.__version__) >= version.parse("2.8"):
        logging.info("ComfyUI-GGUF: Allowing full torch compile")
        return dummy_decorator # torch compile works
    if chained_hasattr(torch, "_dynamo.config.nontraceable_tensor_subclasses"):
        logging.info("ComfyUI-GGUF: Allowing full torch compile (nightly)")
        return dummy_decorator # torch compile works, nightly before 2.8 release
    else:
        logging.info("ComfyUI-GGUF: Partial torch compile only, consider updating pytorch")
        return torch.compiler.disable

torch_compiler_disable = get_torch_compiler_disable_decorator()


class GGUFQuantizedLayout(QuantizedLayout):
    @dataclass(frozen=True)
    class Params:
        orig_dtype: torch.dtype
        orig_shape: tuple[int, ...]
        qtype: gguf.GGMLQuantizationType
        dequant_dtype: torch.dtype | str | None = None

        def to_device(self, device):
            return self

        def clone(self):
            return self

        def copy_from(self, src, non_blocking=False):
            if self.orig_shape != src.orig_shape or self.qtype != src.qtype:
                raise ValueError("Cannot copy between GGUF tensors with different shapes or quantization types")

    @classmethod
    def quantize(cls, tensor, **kwargs):
        raise NotImplementedError("GGUF weights are wrapped without requantization")

    @classmethod
    def dequantize(cls, qdata, params):
        dtype = params.orig_dtype if params.dequant_dtype == "target" else params.dequant_dtype
        result = dequantize(qdata, params.qtype, params.orig_shape, dtype=dtype)
        return result.to(params.orig_dtype)

    @classmethod
    def get_plain_tensors(cls, qtensor):
        return (qtensor._qdata,)

    @classmethod
    def state_dict_tensors(cls, qdata, params):
        return {"weight": qdata}

    @classmethod
    def dequantize_embedding(cls, qtensor, indices, out_dtype=None):
        params = qtensor._params
        vocab_size, embedding_dim = params.orig_shape
        block_size, type_size = gguf.GGML_QUANT_SIZES[params.qtype]
        if embedding_dim % block_size != 0:
            raise ValueError(f"GGUF embedding row size {embedding_dim} is not aligned to {block_size}")

        row_size = embedding_dim // block_size * type_size
        qdata = qtensor._qdata.reshape(-1).view(torch.uint8)
        rows = qdata.reshape(vocab_size, row_size)
        row_indices = indices.reshape(-1).to(device=qdata.device, dtype=torch.long)
        selected = rows.index_select(0, row_indices)
        if selected.device != indices.device:
            selected = selected.to(indices.device, non_blocking=True)

        dtype = params.orig_dtype if params.dequant_dtype == "target" else params.dequant_dtype
        result = dequantize(selected, params.qtype, (row_indices.numel(), embedding_dim), dtype=dtype)
        result = result.to(params.orig_dtype).reshape(*indices.shape, embedding_dim)
        return result.to(dtype=out_dtype) if out_dtype is not None else result


register_layout_class(GGUF_QUANTIZED_LAYOUT, GGUFQuantizedLayout)


def is_gguf_quantized_tensor(tensor):
    return isinstance(tensor, QuantizedTensor) and tensor._layout_cls == GGUF_QUANTIZED_LAYOUT


def make_gguf_quantized_tensor(tensor, dtype, dequant_dtype=None):
    qdata = tensor.as_subclass(torch.Tensor)
    params = GGUFQuantizedLayout.Params(
        orig_dtype=dtype,
        orig_shape=tuple(tensor.tensor_shape),
        qtype=tensor.tensor_type,
        dequant_dtype=dequant_dtype,
    )
    return QuantizedTensor(qdata, GGUF_QUANTIZED_LAYOUT, params)

class GGMLTensor(torch.Tensor):
    """
    Main tensor-like class for storing quantized weights
    """
    def __init__(self, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches

    def __new__(cls, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        new.patches = getattr(self, "patches", []).copy()
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        # fixes .weight.copy_ in comfy/clip_model/CLIPTextModel
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            logging.warning(f"ignoring 'copy_' on tensor: {e}")

    def new_empty(self, size, *args, **kwargs):
        # Intel Arc fix, ref#50
        new_tensor = super().new_empty(size, *args, **kwargs)
        return GGMLTensor(
                new_tensor,
                tensor_type = getattr(self, "tensor_type", None),
                tensor_shape = size,
                patches = getattr(self, "patches", []).copy()
        )

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape

class GGMLLayer(torch.nn.Module):
    """
    This (should) be responsible for de-quantizing on the fly
    """
    comfy_cast_weights = True
    dequant_dtype = None
    patch_dtype = None
    largest_layer = False
    torch_compatible_tensor_types = {None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}

    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight, bias = state_dict.get(f"{prefix}weight"), state_dict.get(f"{prefix}bias")
        # NOTE: using modified load for linear due to not initializing on creation, see GGMLOps todo
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(self, torch.nn.Linear):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # Not strictly required, but fixes embedding shape mismatch. Threshold set in loader.py
        if isinstance(self, torch.nn.Embedding) and self.weight.shape[0] >= (64 * 1024):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        prefix_len = len(prefix)
        for k,v in state_dict.items():
            if k[prefix_len:] == "weight":
                if isinstance(v, GGMLTensor) and is_quantized(v):
                    dtype = getattr(self, "weight_comfy_model_dtype", None) or torch.float16
                    v = make_gguf_quantized_tensor(v, dtype, self.dequant_dtype)
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == "bias" and v is not None:
                if isinstance(v, GGMLTensor) and is_quantized(v):
                    dtype = getattr(self, "bias_comfy_model_dtype", None) or torch.float16
                    v = make_gguf_quantized_tensor(v, dtype, self.dequant_dtype)
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                unexpected_keys.append(k)

        # For Linear layer with missing weight
        if self.weight is None and isinstance(self, torch.nn.Linear):
            v = torch.zeros(self.in_features, self.out_features)
            self.weight = torch.nn.Parameter(v, requires_grad=False)
            missing_keys.append(prefix+"weight")

        # for vram estimation (TODO: less fragile logic?)
        if getattr(self.weight, "is_largest_weight", False):
            self.largest_layer = True

    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        # This is a fake state dict for vram estimation
        if is_gguf_quantized_tensor(self.weight):
            weight = torch.empty(self.weight.shape, device=torch.device("meta"), dtype=self.weight.dtype)
        else:
            weight = torch.zeros_like(self.weight, device=torch.device("meta"))
        destination[prefix + "weight"] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device("meta"))
            destination[prefix + "bias"] = bias

        # Take into account space required for dequantizing the largest tensor
        if self.largest_layer:
            shape = getattr(self.weight, "tensor_shape", self.weight.shape)
            dtype = self.dequant_dtype if self.dequant_dtype and self.dequant_dtype != "target" else torch.float16
            temp = torch.empty(*shape, device=torch.device("meta"), dtype=dtype)
            destination[prefix + "temp.weight"] = temp

        return
        # This would return the dequantized state dict
        destination[prefix + "weight"] = self.get_weight(self.weight)
        if bias is not None:
            destination[prefix + "bias"] = self.get_weight(self.bias)

    def get_weight(self, tensor, dtype):
        if tensor is None:
            return

        # consolidate and load patches to GPU in async
        patch_list = []
        device = tensor.device
        for patches, key in getattr(tensor, "patches", []):
            patch_list += move_patch_to_device(patches, device)

        # dequantize tensor while patches load
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)

        # prevent propagating custom tensor class
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight)

        # apply patches
        if len(patch_list) > 0:
            if self.patch_dtype is None:
                weight = comfy.lora.calculate_weight(patch_list, weight, key)
            else:
                # for testing, may degrade image quality
                patch_dtype = dtype if self.patch_dtype == "target" else self.patch_dtype
                weight = comfy.lora.calculate_weight(patch_list, weight, key, patch_dtype)
        return weight

    @torch_compiler_disable()
    def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, "dtype", torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device
        if dtype is None:
            dtype = getattr(s, "weight_comfy_model_dtype", None) or s.weight.dtype
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = s.weight.device

        cast_dtype = s.weight.dtype if is_gguf_quantized_tensor(s.weight) else dtype
        weight, bias, offload_state = comfy.ops.cast_bias_weight(
            s,
            input=input,
            dtype=cast_dtype,
            device=device,
            bias_dtype=bias_dtype,
            offloadable=True,
        )
        try:
            weight = s.get_weight(weight, dtype)
            bias = s.get_weight(bias, bias_dtype)
        except Exception:
            comfy.ops.uncast_bias_weight(s, weight, bias, offload_state)
            raise
        return weight, bias, offload_state

    @contextlib.contextmanager
    def cast_bias_weight_context(s, *args, **kwargs):
        weight, bias, offload_state = s.cast_bias_weight(*args, **kwargs)
        try:
            yield weight, bias
        finally:
            comfy.ops.uncast_bias_weight(s, weight, bias, offload_state)

    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.weight is None:
            out = super().forward_comfy_cast_weights(input, *args, **kwargs)
        else:
            out = self.forward_ggml_cast_weights(input, *args, **kwargs)

        # non-ggml forward might still propagate custom tensor class
        if isinstance(out, GGMLTensor):
            out = torch.Tensor(out)
        return out

    def forward_ggml_cast_weights(self, input):
        raise NotImplementedError

class GGMLOps(comfy.ops.manual_cast):
    """
    Dequantize weights on the fly before doing the compute
    """
    class Linear(GGMLLayer, comfy.ops.manual_cast.Linear):
        gguf_compute_mode = "ggml"

        def _warn_tensor_core_fallback(self, reason):
            try:
                from . import tensor_core
                qtype = getattr(getattr(getattr(self, "weight", None), "_params", None), "qtype", None)
                qtype_name = getattr(qtype, "name", "unknown")
                tensor_core.warn_once((qtype_name, self.gguf_compute_mode, reason), reason)
            except ImportError:
                logging.warning("ComfyUI-GGUF Tensor Core fallback: %s", reason)

        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            # TODO: better workaround for reserved memory spike on windows
            # Issue is with `torch.empty` still reserving the full memory for the layer
            # Windows doesn't over-commit memory so without this 24GB+ of pagefile is used
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None
            self.weight_function = []
            self.bias_function = []
            self.weight_comfy_model_dtype = dtype
            self.bias_comfy_model_dtype = dtype

        def forward_ggml_cast_weights(self, input):
            if getattr(self, "gguf_compute_mode", "ggml") == "ggml":
                fast = self._try_ggml_mmq_linear(input)
            else:
                fast = self._try_tensor_core_linear(input)
            if fast is not None:
                return fast
            with self.cast_bias_weight_context(input) as (weight, bias):
                return torch.nn.functional.linear(input, weight, bias)

        @torch_compiler_disable()
        def _try_ggml_mmq_linear(self, input):
            if not is_gguf_quantized_tensor(self.weight):
                return None
            if getattr(self, "dequant_dtype", None) is not None:
                self._warn_ggml_fallback("explicit dequant_dtype requires the original compute path")
                return None
            if getattr(self, "comfy_force_cast_weights", False):
                self._warn_ggml_fallback("ComfyUI forced weight casting")
                return None
            if getattr(self, "weight_function", []) or getattr(self, "bias_function", []):
                self._warn_ggml_fallback("LoRA or weight hooks are active")
                return None
            if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                self._warn_ggml_fallback(f"unsupported activation dtype {input.dtype}")
                return None
            if input.device.type != "cuda":
                self._warn_ggml_fallback(f"unsupported execution device {input.device}")
                return None

            try:
                from . import ggml_backend

                ggml_backend.ensure_device(input.device)
                qtype = self.weight._params.qtype
                qtype_name = getattr(qtype, "name", "")
                if qtype_name not in ggml_backend.SUPPORTED_WEIGHT_TYPES:
                    self._warn_ggml_fallback(f"unsupported quantization type {qtype_name or 'unknown'}")
                    return None
                qtype_id = int(getattr(qtype, "value", qtype))
            except (AttributeError, TypeError, ValueError) as exc:
                self._warn_ggml_fallback(f"invalid GGUF quantization metadata: {exc}")
                return None

            in_features = int(self.in_features)
            out_features = int(self.out_features)
            if tuple(self.weight._params.orig_shape) != (out_features, in_features):
                self._warn_ggml_fallback("GGUF matrix shape does not match the Linear module")
                return None
            if input.shape[-1] != in_features or not input.is_contiguous():
                self._warn_ggml_fallback("activation layout or K dimension is unsupported")
                return None

            try:
                with comfy.ops.CastBiasWeightContext(
                    self,
                    input=None,
                    dtype=self.weight.dtype,
                    device=input.device,
                    bias_dtype=input.dtype,
                    offloadable=True,
                ) as (weight, bias):
                    if not is_gguf_quantized_tensor(weight):
                        self._warn_ggml_fallback("ComfyUI cast did not preserve the packed GGUF tensor")
                        return None
                    qdata = weight._qdata
                    if qdata.device != input.device or not qdata.is_contiguous():
                        self._warn_ggml_fallback("packed GGUF weights are not contiguous on the execution device")
                        return None

                    input_2d = input.view(-1, in_features)
                    try:
                        output_2d = ggml_backend.linear_mmq(qdata, input_2d, out_features, qtype_id)
                    except ggml_backend.GGMLMMQUnsupported as exc:
                        self._warn_ggml_fallback(str(exc))
                        return None
                    if bias is not None:
                        output_2d.add_(bias)
                    output = output_2d.reshape(*input.shape[:-1], out_features)
                    ggml_backend.info_once(
                        (qtype_name, str(input.device), str(input.dtype)),
                        f"ggml MMQ {qtype_name}, Q8_1 activations, input={input.dtype}, device={input.device}; "
                        f"pinned ggml {ggml_backend.EXPECTED_GGML_COMMIT}",
                    )
                    return output
            except (AttributeError, IndexError, TypeError) as exc:
                self._warn_ggml_fallback(f"GGUF layout is unsupported: {exc}")
                return None

        def _warn_ggml_fallback(self, reason):
            try:
                from . import ggml_backend

                qtype = getattr(getattr(getattr(self, "weight", None), "_params", None), "qtype", None)
                qtype_name = getattr(qtype, "name", "unknown")
                ggml_backend.warn_once((qtype_name, reason), reason)
            except ImportError:
                logging.warning("ComfyUI-GGUF ggml MMQ fallback: %s", reason)

        def _try_tensor_core_linear(self, input):
            mode = getattr(self, "gguf_compute_mode", "ggml")
            if mode == "original" or mode not in ("mixed", "int8"):
                return None
            if not is_gguf_quantized_tensor(self.weight):
                return None
            if getattr(self, "dequant_dtype", None) is not None:
                self._warn_tensor_core_fallback("explicit dequant_dtype requires the original compute path")
                return None
            if getattr(self, "comfy_force_cast_weights", False):
                self._warn_tensor_core_fallback("ComfyUI forced weight casting")
                return None
            if getattr(self, "weight_function", []) or getattr(self, "bias_function", []):
                self._warn_tensor_core_fallback("LoRA or weight hooks are active")
                return None
            if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                self._warn_tensor_core_fallback(f"unsupported activation dtype {input.dtype}")
                return None

            qtype = getattr(self.weight._params.qtype, "name", "")
            if qtype not in ("Q4_K", "Q5_K", "Q6_K", "IQ3_S", "IQ3_XXS"):
                self._warn_tensor_core_fallback(f"unsupported quantization type {qtype or 'unknown'}")
                return None
            weight_bits = 4 if mode == "mixed" and qtype in ("Q4_K", "IQ3_S") else 8
            if weight_bits == 4 and qtype not in ("Q4_K", "IQ3_S"):
                return None

            in_features = int(self.in_features)
            out_features = int(self.out_features)
            if tuple(self.weight._params.orig_shape) != (out_features, in_features):
                self._warn_tensor_core_fallback("GGUF matrix shape does not match the Linear module")
                return None
            if input.shape[-1] != in_features or in_features % 256:
                self._warn_tensor_core_fallback("Linear K dimension is not GGUF block aligned")
                return None
            try:
                from . import tensor_core
                # CastBiasWeightContext performs ComfyUI's normal device transfer,
                # DynamicVRAM pinning, and stream lifetime management while keeping
                # the GGUF QuantizedTensor compressed.
                with comfy.ops.CastBiasWeightContext(
                    self,
                    input=None,
                    dtype=self.weight.dtype,
                    device=input.device,
                    bias_dtype=input.dtype,
                    offloadable=True,
                ) as (weight, bias):
                    if not is_gguf_quantized_tensor(weight):
                        return None
                    qdata = weight._qdata
                    if qdata.device != input.device or not qdata.is_contiguous():
                        self._warn_tensor_core_fallback("packed weight is not contiguous on the execution device")
                        return None
                    input_2d = input.reshape(-1, in_features).contiguous()
                    output_2d = torch.empty(
                        (input_2d.shape[0], out_features),
                        device=input.device,
                        dtype=input.dtype,
                    )
                    try:
                        tensor_core.launch_linear(
                            qdata, input_2d, bias, output_2d, qtype, weight_bits
                        )
                        tensor_core.info_once(
                            (qtype, weight_bits, str(input.device), str(input.dtype)),
                            f"{qtype} W{weight_bits}A{weight_bits} on {input.device} ({input.dtype})",
                        )
                    except (RuntimeError, OSError) as exc:
                        # Compilation/setup errors are safe to fall back before a
                        # kernel is enqueued. Driver launch errors are surfaced.
                        if "cuLaunchKernel" in str(exc):
                            raise
                        tensor_core.warn_once(
                            (qtype, weight_bits, str(input.device)),
                            f"{type(exc).__name__}: {exc}",
                        )
                        return None
                return output_2d.reshape(*input.shape[:-1], out_features)
            except (AttributeError, IndexError, TypeError, ValueError) as exc:
                # Shape/dtype/layout incompatibilities use the established path.
                if "tensor_core" in locals():
                    tensor_core.warn_once(
                        (qtype, weight_bits, "layout"),
                        f"{type(exc).__name__}: {exc}",
                    )
                return None

    class Conv2d(GGMLLayer, comfy.ops.manual_cast.Conv2d):
        def forward_ggml_cast_weights(self, input):
            with self.cast_bias_weight_context(input) as (weight, bias):
                return self._conv_forward(input, weight, bias)

    class Embedding(GGMLLayer, comfy.ops.manual_cast.Embedding):
        def host_rows(self, input, out_dtype=None):
            if is_gguf_quantized_tensor(self.weight):
                return GGUFQuantizedLayout.dequantize_embedding(self.weight, input, out_dtype)
            return super().host_rows(input, out_dtype=out_dtype)

        def forward_ggml_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16:
                out_dtype = None
            if (is_gguf_quantized_tensor(self.weight) and self.max_norm is None
                    and not getattr(self, "weight_function", []) and not hasattr(self, "_prefetch")):
                return self.host_rows(input, out_dtype=output_dtype)
            with self.cast_bias_weight_context(device=input.device, dtype=out_dtype) as (weight, _bias):
                return torch.nn.functional.embedding(
                    input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
                ).to(dtype=output_dtype)

    class LayerNorm(GGMLLayer, comfy.ops.manual_cast.LayerNorm):
        def forward_ggml_cast_weights(self, input):
            if self.weight is None:
                return super().forward_comfy_cast_weights(input)
            with self.cast_bias_weight_context(input) as (weight, bias):
                return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

    class GroupNorm(GGMLLayer, comfy.ops.manual_cast.GroupNorm):
        def forward_ggml_cast_weights(self, input):
            with self.cast_bias_weight_context(input) as (weight, bias):
                return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)


def make_gguf_ops(compute_mode="ggml"):
    """Create isolated custom operations for one GGUF loader choice."""
    if compute_mode not in ("mixed", "int8", "ggml", "original"):
        raise ValueError(f"Unknown GGUF compute mode: {compute_mode}")
    if compute_mode == "ggml":
        from . import ggml_backend

        ggml_backend.ensure_backend()
    linear = type(
        f"GGUFLinear_{compute_mode}",
        (GGMLOps.Linear,),
        {"gguf_compute_mode": compute_mode, "__module__": __name__},
    )
    return type(
        f"GGMLOps_{compute_mode}",
        (GGMLOps,),
        {"Linear": linear, "__module__": __name__},
    )

def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(move_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [move_patch_to_device(x, device) for x in item]
    else:
        return item
