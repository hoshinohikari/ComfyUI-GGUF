#include <cmath>

namespace {

std::mutex ggml_cuda_mmq_external_mutex;
struct ggml_cuda_mmq_context_holder {
    int device;
    cudaStream_t borrowed_stream;
    std::unique_ptr<ggml_backend_cuda_context> context;

    ggml_cuda_mmq_context_holder(int device, cudaStream_t stream) :
            device(device), borrowed_stream(stream == nullptr ? cudaStreamLegacy : stream),
            context(std::make_unique<ggml_backend_cuda_context>(device)) {
        context->streams[device][0] = borrowed_stream;
    }

    ~ggml_cuda_mmq_context_holder() {
        // The ggml context owns its pool but the stream belongs to PyTorch.
        context->streams[device][0] = nullptr;
    }
};

std::map<std::pair<int, uintptr_t>, std::unique_ptr<ggml_cuda_mmq_context_holder>> ggml_cuda_mmq_external_contexts;

ggml_backend_cuda_context & ggml_cuda_mmq_external_context(int device, cudaStream_t stream) {
    const auto key = std::make_pair(device, reinterpret_cast<uintptr_t>(stream));
    auto it = ggml_cuda_mmq_external_contexts.find(key);
    if (it == ggml_cuda_mmq_external_contexts.end()) {
        it = ggml_cuda_mmq_external_contexts.emplace(
            key, std::make_unique<ggml_cuda_mmq_context_holder>(device, stream)).first;
    }
    return *it->second->context;
}

void ggml_cuda_mmq_external_set_error(char * error, size_t error_capacity, const char * message) {
    if (error && error_capacity) {
        std::snprintf(error, error_capacity, "%s", message ? message : "unknown error");
    }
}

bool ggml_cuda_mmq_external_supported_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_mmq_external_supported_glu(ggml_glu_op op) {
    switch (op) {
        case GGML_GLU_OP_GEGLU:
        case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_SWIGLU_OAI:
        case GGML_GLU_OP_SWIGLU_CLAMP:
            return true;
        default:
            return false;
    }
}

int ggml_cuda_mmq_external_fusion_status(
        int device,
        ggml_type type,
        int64_t rows,
        int64_t columns,
        ggml_glu_op glu_op,
        float glu_limit,
        char * error,
        size_t error_capacity) {
    if (!ggml_cuda_mmq_external_supported_glu(glu_op) ||
            (glu_op == GGML_GLU_OP_SWIGLU_CLAMP && (!std::isfinite(glu_limit) || glu_limit <= 0.0f)) ||
            (glu_op != GGML_GLU_OP_SWIGLU_CLAMP && glu_limit != 0.0f)) {
        ggml_cuda_mmq_external_set_error(error, error_capacity, "unsupported GGML CUDA GLU operation or limit");
        return 1;
    }
    if (!ggml_cuda_mmq_fusion_supported(type) || ggml_cuda_mmq_needs_fallback(columns)) {
        ggml_cuda_mmq_external_set_error(error, error_capacity, "GGML CUDA MMQ does not support fusion for this weight type or shape");
        return 2;
    }

    const int cc = ggml_cuda_info().devices[device].cc;
    const size_t smpbo = ggml_cuda_info().devices[device].smpbo;
    const ggml_prec precision = ggml_cuda_mmq_get_prec_src1(type, GGML_PREC_Q8, cc);
    if (ggml_cuda_mmq_get_J_best(type, rows, false, cc, smpbo, precision) <= 0) {
        ggml_cuda_mmq_external_set_error(error, error_capacity, "GGML CUDA MMQ gate/up fusion has no tile for this shape and device");
        return 2;
    }
    return 0;
}

} // namespace

extern "C" int ggml_cuda_mmq_external_workspace_bytes(
        int device,
        int type_id,
        int64_t rows,
        int64_t columns,
        int64_t inner,
        int fused,
        int glu_op_id,
        float glu_limit,
        size_t * workspace_bytes,
        char * error,
        size_t error_capacity) {
    if (workspace_bytes) *workspace_bytes = 0;
    const ggml_type type = static_cast<ggml_type>(type_id);
    if (!workspace_bytes || rows <= 0 || columns <= 0 || inner <= 0 ||
            !ggml_cuda_mmq_external_supported_type(type) || inner % ggml_blck_size(type) != 0) {
        ggml_cuda_mmq_external_set_error(error, error_capacity, "unsupported GGUF MMQ type or matrix shape");
        return 1;
    }
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        ggml_cuda_mmq_external_set_error(error, error_capacity, "CUDA device index is out of range");
        return 1;
    }

    const bool use_fusion = fused != 0;
    const ggml_glu_op glu_op = static_cast<ggml_glu_op>(glu_op_id);
    if (use_fusion) {
        const int fusion_status = ggml_cuda_mmq_external_fusion_status(
            device, type, rows, columns, glu_op, glu_limit, error, error_capacity);
        if (fusion_status != 0) {
            return fusion_status;
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;
    const size_t smpbo = ggml_cuda_info().devices[device].smpbo;
    const int nsm = ggml_cuda_info().devices[device].nsm;
    const bool fallback = use_fusion ? false : ggml_cuda_mmq_needs_fallback(columns);
    // MMQ activation precision is independent of the FP32 staging tensor dtype.
    const ggml_prec precision = ggml_cuda_mmq_get_prec_src1(type, GGML_PREC_Q8, cc);
    int nthreads = 0;
    const int J = ggml_cuda_mmq_get_J_best(type, rows, fallback, cc, smpbo, precision, &nthreads);
    if (J <= 0) {
        ggml_cuda_mmq_external_set_error(error, error_capacity, "ggml MMQ has no tile configuration for this shape and format");
        return 2;
    }

    const int64_t inner_padded = GGML_PAD(inner, MATRIX_ROW_PADDING);
    const size_t load_chunk_size = nthreads * sizeof(int);
    const size_t activation_padding =
        ((J * sizeof(block_q8_1_mmq) + load_chunk_size - 1) / load_chunk_size) * load_chunk_size;
    size_t required = static_cast<size_t>(rows) * inner_padded * sizeof(block_q8_1_mmq) / QK8_1_MMQ
        + activation_padding;

    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, precision);
    if (config.stream_k && !use_fusion) {
        const int nti = static_cast<int>((columns + config.I - 1) / config.I);
        const int ntj = static_cast<int>((rows + J - 1) / J);
        const int ntiles = ntj * nti;
        const int tile_waves = (ntiles + nsm - 1) / nsm;
        const int efficiency = 100 * ntiles / (nsm * tile_waves);
        const int nwork = ntiles * static_cast<int>((inner + config.K_vram - 1) / config.K_vram);
        const int blocks = GGML_CUDA_CC_IS_NVIDIA(cc) && efficiency >= 90
            ? ntiles : std::min(nsm, nwork);
        if (ntiles % blocks != 0) {
            required += static_cast<size_t>(blocks) * config.J * config.I * sizeof(float);
        }
    }

    *workspace_bytes = required;
    return 0;
}

// Weight, activation, and output pointers are borrowed from the caller. ggml owns only metadata and scratch.
static int ggml_cuda_mmq_external_impl(
        int device,
        const void * up_weight_data,
        const void * gate_weight_data,
        int type_id,
        const void * input_data,
        int input_type_id,
        int output_type_id,
        int glu_op_id,
        float glu_limit,
        void * output_data,
        int64_t rows,
        int64_t columns,
        int64_t inner,
        void * caller_stream_ptr,
        void ** compute_stream_ptr,
        char * error,
        size_t error_capacity) {
    if (compute_stream_ptr) *compute_stream_ptr = nullptr;
    const bool fused = gate_weight_data != nullptr;
    if (!up_weight_data || !input_data || !output_data ||
            rows <= 0 || columns <= 0 || inner <= 0) {
        ggml_cuda_mmq_external_set_error(error, error_capacity, "invalid null pointer or matrix shape");
        return 1;
    }

    const ggml_type type = static_cast<ggml_type>(type_id);
    const ggml_type input_type = static_cast<ggml_type>(input_type_id);
    const ggml_type output_type = static_cast<ggml_type>(output_type_id);
    if (!ggml_cuda_mmq_external_supported_type(type)) {
        ggml_cuda_mmq_external_set_error(error, error_capacity, "unsupported GGUF quantization type");
        return 1;
    }
    if (input_type != GGML_TYPE_F32 && input_type != GGML_TYPE_F16 && input_type != GGML_TYPE_BF16) {
        ggml_cuda_mmq_external_set_error(error, error_capacity, "unsupported activation dtype for GGML MMQ");
        return 1;
    }
    if (output_type != GGML_TYPE_F32 &&
            !(output_type == GGML_TYPE_BF16 && !fused && type == GGML_TYPE_Q8_0 && input_type == GGML_TYPE_BF16)) {
        ggml_cuda_mmq_external_set_error(error, error_capacity, "BF16 direct output is supported only for Q8_0 with BF16 activations");
        return 1;
    }
    const ggml_glu_op glu_op = static_cast<ggml_glu_op>(glu_op_id);
    if (fused && (!ggml_cuda_mmq_external_supported_glu(glu_op) ||
            (glu_op == GGML_GLU_OP_SWIGLU_CLAMP && (!std::isfinite(glu_limit) || glu_limit <= 0.0f)) ||
            (glu_op != GGML_GLU_OP_SWIGLU_CLAMP && glu_limit != 0.0f))) {
        ggml_cuda_mmq_external_set_error(error, error_capacity, "unsupported GGML CUDA GLU operation or limit");
        return 1;
    }
    if (inner % ggml_blck_size(type) != 0) {
        ggml_cuda_mmq_external_set_error(error, error_capacity, "K dimension is not aligned to the GGML quantization block");
        return 1;
    }
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        ggml_cuda_mmq_external_set_error(error, error_capacity, "CUDA device index is out of range");
        return 1;
    }

    std::lock_guard<std::mutex> lock(ggml_cuda_mmq_external_mutex);
    ggml_cuda_set_device(device);

    cudaStream_t caller_stream = reinterpret_cast<cudaStream_t>(caller_stream_ptr);
    auto & backend_context = ggml_cuda_mmq_external_context(device, caller_stream);
    cudaStream_t compute_stream = backend_context.stream();
    if (compute_stream_ptr) *compute_stream_ptr = reinterpret_cast<void *>(compute_stream);

    ggml_init_params params{};
    params.mem_size = 64 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * metadata = ggml_init(params);
    if (!metadata) {
        ggml_cuda_mmq_external_set_error(error, error_capacity, "failed to allocate ggml tensor metadata");
        return 3;
    }

    int result = 0;
    try {
        ggml_tensor * src0 = ggml_new_tensor_2d(metadata, type, inner, columns);
        ggml_tensor * src1 = ggml_new_tensor_2d(metadata, input_type, inner, rows);
        if (!src0 || !src1) {
            throw std::runtime_error("failed to allocate ggml tensor metadata");
        }
        ggml_tensor * gate = fused ? ggml_new_tensor_2d(metadata, type, inner, columns) : nullptr;
        if (fused && !gate) {
            throw std::runtime_error("failed to allocate GGML gate tensor metadata");
        }
        ggml_tensor * up_mat = ggml_mul_mat(metadata, src0, src1);
        ggml_tensor * gate_mat = fused ? ggml_mul_mat(metadata, gate, src1) : nullptr;
        ggml_tensor * dst = fused
            ? ggml_glu_split(metadata, gate_mat, up_mat, glu_op)
            : up_mat;
        if (!dst || dst->type != GGML_TYPE_F32 || dst->ne[0] != columns || dst->ne[1] != rows) {
            throw std::runtime_error("ggml produced incompatible MMQ tensor metadata");
        }
        if (output_type == GGML_TYPE_BF16) {
            dst->type = GGML_TYPE_BF16;
            dst->nb[0] = ggml_type_size(dst->type);
            dst->nb[1] = ggml_row_size(dst->type, dst->ne[0]);
            dst->nb[2] = dst->nb[1] * dst->ne[1];
            dst->nb[3] = dst->nb[2] * dst->ne[2];
        }

        src0->data = const_cast<void *>(up_weight_data);
        if (fused) {
            gate->data = const_cast<void *>(gate_weight_data);
        }
        src1->data = const_cast<void *>(input_data);
        dst->data = output_data;
        if (fused) {
            ggml_prec_set_src(up_mat, GGML_PREC_Q8, 1);
            ggml_prec_set_src(gate_mat, GGML_PREC_Q8, 1);
            if (glu_op == GGML_GLU_OP_SWIGLU_CLAMP) {
                ggml_set_op_params_f32(dst, 3, glu_limit);
            } else if (glu_op == GGML_GLU_OP_SWIGLU_OAI) {
                ggml_set_op_params_f32(dst, 2, 1.702f);
                ggml_set_op_params_f32(dst, 3, 7.0f);
            }
        } else {
            ggml_prec_set_src(dst, GGML_PREC_Q8, 1);
        }

        const int cc = ggml_cuda_info().devices[device].cc;
        const size_t smpbo = ggml_cuda_info().devices[device].smpbo;
        const bool fallback = ggml_cuda_mmq_needs_fallback(src0->ne[1]);
        const ggml_prec precision = ggml_cuda_mmq_get_prec_src1(type, GGML_PREC_Q8, cc);
        const bool no_mmq_tile = ggml_cuda_mmq_get_J_best(
            type, rows, fused ? false : fallback, cc, smpbo, precision) <= 0;
        const bool invalid_fused_shape = fused && (
            !ggml_cuda_mmq_fusion_supported(type) ||
            fallback ||
            no_mmq_tile ||
            !ggml_cuda_mmq_fusion_is_efficient(type, GGML_PREC_Q8, columns, rows, 1, device));
        if (invalid_fused_shape) {
            ggml_cuda_mmq_external_set_error(error, error_capacity,
                "GGML CUDA MMQ gate/up fusion is unsupported for this qtype, shape, GLU operation, or device");
            result = 2;
        } else if (no_mmq_tile) {
            ggml_cuda_mmq_external_set_error(error, error_capacity, "ggml MMQ has no tile configuration for this shape and format");
            result = 2;
        } else {
            if (fused) {
                ggml_cuda_mm_fusion_args_host fusion{};
                fusion.gate = gate;
                fusion.glu_op = glu_op;
                fusion.glu_limit = glu_op == GGML_GLU_OP_SWIGLU_OAI ? 7.0f : glu_limit;
                fusion.prec_src1 = GGML_PREC_Q8;
                ggml_cuda_mul_mat_q(backend_context, src0, src1, nullptr, dst, &fusion);
            } else {
                // An ordinary Linear has neither a fused gate nor an NVFP4 scale.
                ggml_cuda_mul_mat_q(backend_context, src0, src1, nullptr, dst, nullptr);
            }
        }
    } catch (const std::exception & exception) {
        ggml_cuda_mmq_external_set_error(error, error_capacity, exception.what());
        result = 3;
    } catch (...) {
        ggml_cuda_mmq_external_set_error(error, error_capacity, "unknown ggml MMQ bridge failure");
        result = 3;
    }

    ggml_free(metadata);
    return result;
}

extern "C" int ggml_cuda_mmq_external(
        int device,
        const void * weight_data,
        int type_id,
        const void * input_data,
        int input_type_id,
        int output_type_id,
        void * output_data,
        int64_t rows,
        int64_t columns,
        int64_t inner,
        void * caller_stream_ptr,
        void ** compute_stream_ptr,
        char * error,
        size_t error_capacity) {
    return ggml_cuda_mmq_external_impl(device, weight_data, nullptr, type_id, input_data, input_type_id,
        output_type_id, GGML_GLU_OP_SWIGLU, 0.0f, output_data, rows, columns, inner, caller_stream_ptr,
        compute_stream_ptr, error, error_capacity);
}

extern "C" int ggml_cuda_mmq_external_glu(
        int device,
        const void * up_weight_data,
        const void * gate_weight_data,
        int type_id,
        const void * input_data,
        int input_type_id,
        int glu_op_id,
        float glu_limit,
        int output_type_id,
        void * output_data,
        int64_t rows,
        int64_t columns,
        int64_t inner,
        void * caller_stream_ptr,
        void ** compute_stream_ptr,
        char * error,
        size_t error_capacity) {
    return ggml_cuda_mmq_external_impl(device, up_weight_data, gate_weight_data, type_id, input_data, input_type_id,
        output_type_id, glu_op_id, glu_limit, output_data, rows, columns, inner, caller_stream_ptr,
        compute_stream_ptr, error, error_capacity);
}
