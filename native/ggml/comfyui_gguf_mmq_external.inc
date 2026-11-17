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

} // namespace

extern "C" int ggml_cuda_mmq_external_workspace_bytes(
        int device,
        int type_id,
        int64_t rows,
        int64_t columns,
        int64_t inner,
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

    const int cc = ggml_cuda_info().devices[device].cc;
    const size_t smpbo = ggml_cuda_info().devices[device].smpbo;
    const int nsm = ggml_cuda_info().devices[device].nsm;
    const bool fallback = ggml_cuda_mmq_needs_fallback(columns);
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
    if (config.stream_k) {
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

// Weight, input, and output pointers are borrowed from the caller. ggml owns only metadata and scratch.
extern "C" int ggml_cuda_mmq_external_f32(
        int device,
        const void * weight_data,
        int type_id,
        const float * input_data,
        float * output_data,
        int64_t rows,
        int64_t columns,
        int64_t inner,
        void * caller_stream_ptr,
        void ** compute_stream_ptr,
        char * error,
        size_t error_capacity) {
    if (compute_stream_ptr) *compute_stream_ptr = nullptr;
    if (!weight_data || !input_data || !output_data ||
            rows <= 0 || columns <= 0 || inner <= 0) {
        ggml_cuda_mmq_external_set_error(error, error_capacity, "invalid null pointer or matrix shape");
        return 1;
    }

    const ggml_type type = static_cast<ggml_type>(type_id);
    if (!ggml_cuda_mmq_external_supported_type(type)) {
        ggml_cuda_mmq_external_set_error(error, error_capacity, "unsupported GGUF quantization type");
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
        ggml_tensor * src1 = ggml_new_tensor_2d(metadata, GGML_TYPE_F32, inner, rows);
        if (!src0 || !src1) {
            throw std::runtime_error("failed to allocate ggml tensor metadata");
        }
        ggml_tensor * dst = ggml_mul_mat(metadata, src0, src1);
        if (!dst || dst->type != GGML_TYPE_F32 || dst->ne[0] != columns || dst->ne[1] != rows) {
            throw std::runtime_error("ggml produced incompatible MMQ tensor metadata");
        }

        src0->data = const_cast<void *>(weight_data);
        src1->data = const_cast<float *>(input_data);
        dst->data = output_data;
        ggml_prec_set_src(dst, GGML_PREC_Q8, 1);

        const int cc = ggml_cuda_info().devices[device].cc;
        const size_t smpbo = ggml_cuda_info().devices[device].smpbo;
        const bool fallback = ggml_cuda_mmq_needs_fallback(src0->ne[1]);
        const ggml_prec precision = ggml_cuda_mmq_get_prec_src1(type, GGML_PREC_Q8, cc);
        if (ggml_cuda_mmq_get_J_best(type, rows, fallback, cc, smpbo, precision) <= 0) {
            ggml_cuda_mmq_external_set_error(error, error_capacity, "ggml MMQ has no tile configuration for this shape and format");
            result = 2;
        } else {
            // An ordinary Linear has neither a fused gate nor an NVFP4 scale.
            ggml_cuda_mul_mat_q(backend_context, src0, src1, nullptr, dst, nullptr);
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
