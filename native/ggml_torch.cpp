#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <sstream>

#include "ggml.h"

#ifndef COMFYUI_GGML_COMMIT
#define COMFYUI_GGML_COMMIT "unknown"
#endif

#ifndef COMFYUI_TORCH_BUILD
#define COMFYUI_TORCH_BUILD "unknown"
#endif

#ifndef COMFYUI_CUDA_ARCHS
#define COMFYUI_CUDA_ARCHS ""
#endif

#ifndef COMFYUI_BUILD_ID
#define COMFYUI_BUILD_ID "unknown"
#endif

extern "C" int ggml_cuda_mmq_external_workspace_bytes(
    int device,
    int type_id,
    int64_t rows,
    int64_t columns,
    int64_t inner,
    size_t* workspace_bytes,
    char* error,
    size_t error_capacity);

extern "C" int ggml_cuda_mmq_external_f32(
    int device,
    const void* weight_data,
    int type_id,
    const float* input_data,
    float* output_data,
    int64_t rows,
    int64_t columns,
    int64_t inner,
    void* caller_stream,
    void** compute_stream,
    char* error,
    size_t error_capacity);

namespace {

constexpr int64_t kDefaultWorkspaceLimit = 64 * 1024 * 1024;
constexpr int64_t kWorkspaceMargin = 256 * 1024;

bool supported_type(int type_id) {
  switch (static_cast<ggml_type>(type_id)) {
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

std::string bridge_error(const std::array<char, 512>& error) {
  return std::string(error.data());
}

int64_t choose_chunk_rows(
    int device,
    int type_id,
    int64_t remaining,
    int64_t columns,
    int64_t inner,
    int64_t workspace_limit) {
  int64_t low = 0;
  int64_t high = remaining;
  while (low < high) {
    const int64_t candidate = low + (high - low + 1) / 2;
    size_t ggml_workspace = 0;
    std::array<char, 512> error{};
    const int status = ggml_cuda_mmq_external_workspace_bytes(
        device, type_id, candidate, columns, inner, &ggml_workspace,
        error.data(), error.size());
    if (status == 1) {
      throw std::invalid_argument(
          std::string("GGML_MMQ_UNSUPPORTED: ") + bridge_error(error));
    }
    if (status == 2) {
      high = candidate - 1;
      continue;
    }
    if (status != 0) {
      throw std::runtime_error(
          std::string("ggml MMQ workspace planning failed: ") + bridge_error(error));
    }

    const uint64_t torch_staging = static_cast<uint64_t>(candidate) *
        (static_cast<uint64_t>(inner) + static_cast<uint64_t>(columns)) * sizeof(float);
    const uint64_t total = torch_staging + ggml_workspace + kWorkspaceMargin;
    if (total <= static_cast<uint64_t>(workspace_limit)) {
      low = candidate;
    } else {
      high = candidate - 1;
    }
  }
  return low;
}

at::Tensor linear_mmq(
    const at::Tensor& quantized_weight,
    const at::Tensor& input,
    int64_t output_features,
    int64_t type_id,
    int64_t workspace_limit) {
  TORCH_CHECK(input.is_cuda(), "ggml MMQ requires CUDA activations");
  TORCH_CHECK(quantized_weight.is_cuda(), "ggml MMQ requires CUDA GGUF weights");
  TORCH_CHECK(input.device() == quantized_weight.device(), "input and GGUF weights must use the same device");
  TORCH_CHECK(input.dim() == 2, "ggml MMQ expects a two-dimensional activation matrix");
  TORCH_CHECK(input.is_contiguous(), "ggml MMQ requires contiguous activations");
  TORCH_CHECK(quantized_weight.is_contiguous(), "ggml MMQ requires contiguous packed weights");
  TORCH_CHECK(output_features > 0, "output feature count must be positive");
  TORCH_CHECK(workspace_limit > kWorkspaceMargin, "workspace limit is too small");
  TORCH_CHECK(supported_type(static_cast<int>(type_id)), "unsupported GGUF quantization type");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16 ||
          input.scalar_type() == at::kFloat,
      "ggml MMQ accepts FP16, BF16, or FP32 activations");

  const int64_t rows = input.size(0);
  const int64_t inner = input.size(1);
  const int64_t columns = output_features;
  TORCH_CHECK(inner > 0 && columns > 0, "matrix dimensions must be positive");
  if (rows == 0) {
    return at::empty({0, columns}, input.options());
  }
  TORCH_CHECK(
      inner <= std::numeric_limits<int>::max() && columns <= std::numeric_limits<int>::max(),
      "matrix dimensions exceed ggml CUDA MMQ limits");

  auto output = at::empty({rows, columns}, input.options());
  const int device = input.get_device();
  const auto caller_stream = c10::cuda::getCurrentCUDAStream(device);

  int64_t row = 0;
  while (row < rows) {
    const int64_t count = choose_chunk_rows(
        device, static_cast<int>(type_id), rows - row, columns, inner, workspace_limit);
    if (count <= 0) {
      throw std::invalid_argument(
          "GGML_MMQ_UNSUPPORTED: a single row exceeds the 64 MiB temporary workspace budget or has no MMQ tile");
    }

    auto input_f32 = input.narrow(0, row, count).to(at::kFloat).contiguous();
    auto output_f32 = at::empty(
        {count, columns}, input.options().dtype(at::kFloat));

    std::array<char, 512> error{};
    void* compute_stream_ptr = nullptr;
    const int status = ggml_cuda_mmq_external_f32(
        device,
        quantized_weight.data_ptr(),
        static_cast<int>(type_id),
        input_f32.data_ptr<float>(),
        output_f32.data_ptr<float>(),
        count,
        columns,
        inner,
        reinterpret_cast<void*>(caller_stream.stream()),
        &compute_stream_ptr,
        error.data(),
        error.size());

    const auto expected_stream = caller_stream.stream() == nullptr
        ? cudaStreamLegacy
        : caller_stream.stream();
    TORCH_CHECK(
        compute_stream_ptr == reinterpret_cast<void*>(expected_stream),
        "ggml MMQ must execute on the current PyTorch CUDA stream");

    if (status == 1 || status == 2) {
      throw std::invalid_argument(
          std::string("GGML_MMQ_UNSUPPORTED: ") + bridge_error(error));
    }
    TORCH_CHECK(
        status == 0,
        "ggml MMQ CUDA execution failed: ", bridge_error(error));

    output.narrow(0, row, count).copy_(output_f32);
    row += count;
  }
  return output;
}

pybind11::dict build_info() {
  const int runtime_version = CUDART_VERSION;
  pybind11::dict info;
  info["ggml_commit"] = COMFYUI_GGML_COMMIT;
  info["torch_build"] = COMFYUI_TORCH_BUILD;
  info["build_id"] = COMFYUI_BUILD_ID;
  info["cuda_runtime"] = std::to_string(runtime_version / 1000) + "." +
      std::to_string((runtime_version % 1000) / 10);
  pybind11::list architectures;
  std::istringstream arch_stream(COMFYUI_CUDA_ARCHS);
  std::string arch;
  while (std::getline(arch_stream, arch, ',')) {
    architectures.append(std::stoi(arch));
  }
  info["cuda_archs"] = architectures;
  info["workspace_limit"] = kDefaultWorkspaceLimit;
  info["weight_types"] = pybind11::make_tuple(
      "Q1_0", "Q2_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
      "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
      "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL");
  info["weight_type_ids"] = pybind11::make_tuple(
      static_cast<int>(GGML_TYPE_Q1_0),
      static_cast<int>(GGML_TYPE_Q2_0),
      static_cast<int>(GGML_TYPE_Q4_0),
      static_cast<int>(GGML_TYPE_Q4_1),
      static_cast<int>(GGML_TYPE_Q5_0),
      static_cast<int>(GGML_TYPE_Q5_1),
      static_cast<int>(GGML_TYPE_Q8_0),
      static_cast<int>(GGML_TYPE_Q2_K),
      static_cast<int>(GGML_TYPE_Q3_K),
      static_cast<int>(GGML_TYPE_Q4_K),
      static_cast<int>(GGML_TYPE_Q5_K),
      static_cast<int>(GGML_TYPE_Q6_K),
      static_cast<int>(GGML_TYPE_IQ1_S),
      static_cast<int>(GGML_TYPE_IQ2_XXS),
      static_cast<int>(GGML_TYPE_IQ2_XS),
      static_cast<int>(GGML_TYPE_IQ2_S),
      static_cast<int>(GGML_TYPE_IQ3_XXS),
      static_cast<int>(GGML_TYPE_IQ3_S),
      static_cast<int>(GGML_TYPE_IQ4_XS),
      static_cast<int>(GGML_TYPE_IQ4_NL));
  info["activation_type"] = "Q8_1";
  return info;
}

} // namespace

PYBIND11_MODULE(_comfyui_gguf_ggml, module) {
  module.def(
      "linear_mmq",
      &linear_mmq,
      pybind11::arg("quantized_weight"),
      pybind11::arg("input"),
      pybind11::arg("output_features"),
      pybind11::arg("type_id"),
      pybind11::arg("workspace_limit") = kDefaultWorkspaceLimit);
  module.def("build_info", &build_info);
}
