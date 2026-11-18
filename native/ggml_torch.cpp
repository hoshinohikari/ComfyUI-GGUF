#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>

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
    int fused,
    int glu_op_id,
    float glu_limit,
    size_t* workspace_bytes,
    char* error,
    size_t error_capacity);

extern "C" int ggml_cuda_mmq_external(
    int device,
    const void* weight_data,
    int type_id,
    const void* input_data,
    int input_type_id,
    int output_type_id,
    void* output_data,
    int64_t rows,
    int64_t columns,
    int64_t inner,
    void* caller_stream,
    void** compute_stream,
    char* error,
    size_t error_capacity);

extern "C" int ggml_cuda_mmq_external_glu(
    int device,
    const void* up_weight_data,
    const void* gate_weight_data,
    int type_id,
    const void* input_data,
    int input_type_id,
    int glu_op_id,
    float glu_limit,
    int output_type_id,
    void* output_data,
    int64_t rows,
    int64_t columns,
    int64_t inner,
    void* caller_stream,
    void** compute_stream,
    char* error,
    size_t error_capacity);

namespace {

constexpr int64_t kDefaultWorkspaceBudgetBytes = 0;
constexpr int64_t kWorkspaceMargin = 256 * 1024;
constexpr uint64_t kPerfTimedCallsPerShape = 3;

enum class PerfPhase : int { InputPrepare = 0, ActivationQuantAndMmq = 1, OutputCastCopy = 2 };

struct PerfKey {
  int device;
  int type_id;
  int input_dtype;
  int fusion_op;
  int64_t rows;
  int64_t columns;
  int64_t inner;

  bool operator<(const PerfKey& other) const {
    return std::tie(device, type_id, input_dtype, fusion_op, rows, columns, inner) <
        std::tie(other.device, other.type_id, other.input_dtype, other.fusion_op, other.rows, other.columns, other.inner);
  }
};

struct PerfTotals {
  uint64_t calls = 0;
  uint64_t successes = 0;
  uint64_t failures = 0;
  uint64_t chunks = 0;
  uint64_t timed_calls = 0;
  uint64_t direct_output_calls = 0;
  uint64_t output_staging_allocations = 0;
  double plan_cpu_ms = 0.0;
  double input_prepare_gpu_ms = 0.0;
  double activation_quant_and_mmq_gpu_ms = 0.0;
  double output_cast_copy_gpu_ms = 0.0;
};

struct PerfEventPair {
  PerfKey key{};
  PerfPhase phase = PerfPhase::InputPrepare;
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
};

std::mutex g_perf_mutex;
std::map<PerfKey, PerfTotals> g_perf_totals;
std::vector<PerfEventPair> g_perf_events;
uint64_t g_perf_dropped_event_pairs = 0;

void destroy_perf_pair(PerfEventPair& pair) noexcept {
  if (pair.begin) cudaEventDestroy(pair.begin);
  if (pair.end) cudaEventDestroy(pair.end);
  pair.begin = nullptr;
  pair.end = nullptr;
}

class PerfTimer {
 public:
  PerfTimer(bool enabled, bool sample, const PerfKey& key, PerfPhase phase, cudaStream_t stream)
      : enabled_(enabled && sample) {
    if (!enabled_) return;
    pair_.key = key;
    pair_.phase = phase;
    if (cudaEventCreate(&pair_.begin) != cudaSuccess || cudaEventCreate(&pair_.end) != cudaSuccess ||
        cudaEventRecord(pair_.begin, stream) != cudaSuccess) {
      destroy_perf_pair(pair_);
      enabled_ = false;
      std::lock_guard<std::mutex> lock(g_perf_mutex);
      ++g_perf_dropped_event_pairs;
    }
  }

  PerfTimer(const PerfTimer&) = delete;
  PerfTimer& operator=(const PerfTimer&) = delete;

  ~PerfTimer() { destroy_perf_pair(pair_); }

  void finish(cudaStream_t stream) noexcept {
    if (!enabled_) return;
    if (cudaEventRecord(pair_.end, stream) != cudaSuccess) {
      destroy_perf_pair(pair_);
      enabled_ = false;
      std::lock_guard<std::mutex> lock(g_perf_mutex);
      ++g_perf_dropped_event_pairs;
      return;
    }
    try {
      std::lock_guard<std::mutex> lock(g_perf_mutex);
      g_perf_events.push_back(pair_);
      pair_.begin = nullptr;
      pair_.end = nullptr;
      enabled_ = false;
    } catch (...) {
      destroy_perf_pair(pair_);
      enabled_ = false;
      std::lock_guard<std::mutex> lock(g_perf_mutex);
      ++g_perf_dropped_event_pairs;
    }
  }

 private:
  PerfEventPair pair_;
  bool enabled_;
};

class PerfCall {
 public:
  PerfCall(bool enabled, const PerfKey& key, bool direct_output)
      : enabled_(enabled), direct_output_(direct_output), key_(key) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lock(g_perf_mutex);
    auto& totals = g_perf_totals[key_];
    ++totals.calls;
    sample_ = totals.timed_calls < kPerfTimedCallsPerShape;
    if (sample_) ++totals.timed_calls;
  }

  ~PerfCall() {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lock(g_perf_mutex);
    auto& totals = g_perf_totals[key_];
    totals.chunks += chunks_;
    totals.plan_cpu_ms += plan_cpu_ms_;
    totals.output_staging_allocations += output_staging_allocations_;
    if (completed_) {
      ++totals.successes;
      if (direct_output_) ++totals.direct_output_calls;
    }
    else ++totals.failures;
  }

  bool sample() const { return sample_; }
  void add_chunk() { ++chunks_; }
  void add_plan(double plan_ms) { plan_cpu_ms_ += plan_ms; }
  void add_output_staging_allocation() { ++output_staging_allocations_; }
  void complete() { completed_ = true; }

 private:
  bool enabled_;
  bool sample_ = false;
  bool completed_ = false;
  bool direct_output_ = false;
  uint64_t chunks_ = 0;
  uint64_t output_staging_allocations_ = 0;
  double plan_cpu_ms_ = 0.0;
  PerfKey key_;
};

void perf_add_elapsed(std::map<PerfKey, PerfTotals>& totals_by_key, const PerfEventPair& pair, double elapsed_ms) {
  auto& totals = totals_by_key[pair.key];
  switch (pair.phase) {
    case PerfPhase::InputPrepare: totals.input_prepare_gpu_ms += elapsed_ms; break;
    case PerfPhase::ActivationQuantAndMmq: totals.activation_quant_and_mmq_gpu_ms += elapsed_ms; break;
    case PerfPhase::OutputCastCopy: totals.output_cast_copy_gpu_ms += elapsed_ms; break;
  }
}

pybind11::dict collect_performance_stats() {
  std::map<PerfKey, PerfTotals> totals;
  std::vector<PerfEventPair> events;
  uint64_t dropped = 0;
  {
    std::lock_guard<std::mutex> lock(g_perf_mutex);
    totals.swap(g_perf_totals);
    events.swap(g_perf_events);
    dropped = g_perf_dropped_event_pairs;
    g_perf_dropped_event_pairs = 0;
  }

  std::set<int> devices;
  for (const auto& event : events) devices.insert(event.key.device);
  int original_device = 0;
  const bool have_original = cudaGetDevice(&original_device) == cudaSuccess;
  std::string sync_error;
  for (int device : devices) {
    if (cudaSetDevice(device) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
      sync_error = "CUDA synchronization failed while collecting GGUF performance stats";
      break;
    }
  }
  if (have_original) cudaSetDevice(original_device);

  if (sync_error.empty()) {
    for (const auto& event : events) {
      float elapsed_ms = 0.0f;
      if (cudaEventElapsedTime(&elapsed_ms, event.begin, event.end) == cudaSuccess) {
        perf_add_elapsed(totals, event, elapsed_ms);
      } else {
        ++dropped;
      }
    }
  }
  for (auto& event : events) destroy_perf_pair(event);
  TORCH_CHECK(sync_error.empty(), sync_error);

  pybind11::list result;
  for (const auto& [key, value] : totals) {
    pybind11::dict row;
    row["device"] = key.device;
    row["qtype_id"] = key.type_id;
    row["input_dtype"] = c10::toString(static_cast<at::ScalarType>(key.input_dtype));
    row["fusion"] = key.fusion_op < 0 ? "none" : ggml_glu_op_name(static_cast<ggml_glu_op>(key.fusion_op));
    row["M"] = key.rows;
    row["N"] = key.columns;
    row["K"] = key.inner;
    row["calls"] = value.calls;
    row["successful_calls"] = value.successes;
    row["failed_calls"] = value.failures;
    row["chunks"] = value.chunks;
    row["timed_calls"] = value.timed_calls;
    row["direct_output_calls"] = value.direct_output_calls;
    row["output_staging_allocations"] = value.output_staging_allocations;
    row["plan_cpu_ms"] = value.plan_cpu_ms;
    row["input_prepare_gpu_ms_sampled"] = value.input_prepare_gpu_ms;
    row["activation_quant_and_mmq_gpu_ms_sampled"] = value.activation_quant_and_mmq_gpu_ms;
    row["output_cast_copy_gpu_ms_sampled"] = value.output_cast_copy_gpu_ms;
    result.append(row);
  }
  pybind11::dict summary;
  summary["linear_shapes"] = result;
  summary["timing_samples_per_shape_limit"] = kPerfTimedCallsPerShape;
  summary["dropped_event_pairs"] = dropped;
  summary["gpu_synchronized_devices"] = static_cast<int>(devices.size());
  return summary;
}

void reset_performance_stats() {
  std::vector<PerfEventPair> events;
  {
    std::lock_guard<std::mutex> lock(g_perf_mutex);
    g_perf_totals.clear();
    events.swap(g_perf_events);
    g_perf_dropped_event_pairs = 0;
  }
  for (auto& event : events) destroy_perf_pair(event);
}

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
    size_t output_staging_element_size,
    int64_t workspace_budget_bytes,
    bool fused,
    int glu_op_id,
    float glu_limit) {
  if (workspace_budget_bytes == kDefaultWorkspaceBudgetBytes) {
    size_t ggml_workspace = 0;
    std::array<char, 512> error{};
    const int status = ggml_cuda_mmq_external_workspace_bytes(
        device, type_id, remaining, columns, inner, fused ? 1 : 0, glu_op_id, glu_limit, &ggml_workspace,
        error.data(), error.size());
    if (status == 1) {
      throw std::invalid_argument(
          std::string("GGML_MMQ_UNSUPPORTED: ") + bridge_error(error));
    }
    if (status == 2) {
      if (fused) {
        throw std::invalid_argument(
            std::string("GGML_MMQ_UNSUPPORTED: ") + bridge_error(error));
      }
      throw std::runtime_error(
          std::string("GGML MMQ cannot plan the full matrix for this shape: ") + bridge_error(error) +
          ". Set COMFYUI_GGUF_WORKSPACE_BUDGET_MIB to a positive MiB value to enable explicit chunking.");
    }
    if (status != 0) {
      throw std::runtime_error(
          std::string("ggml MMQ workspace planning failed: ") + bridge_error(error));
    }
    return remaining;
  }

  int64_t low = 0;
  int64_t high = remaining;
  bool has_valid_tile = false;
  while (low < high) {
    const int64_t candidate = low + (high - low + 1) / 2;
    size_t ggml_workspace = 0;
    std::array<char, 512> error{};
    const int status = ggml_cuda_mmq_external_workspace_bytes(
        device, type_id, candidate, columns, inner, fused ? 1 : 0, glu_op_id, glu_limit, &ggml_workspace,
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
    has_valid_tile = true;

    const uint64_t torch_output_staging = static_cast<uint64_t>(candidate) *
        static_cast<uint64_t>(columns) * output_staging_element_size;
    const uint64_t total = torch_output_staging + ggml_workspace + kWorkspaceMargin;
    if (total <= static_cast<uint64_t>(workspace_budget_bytes)) {
      low = candidate;
    } else {
      high = candidate - 1;
    }
  }
  if (fused && !has_valid_tile) {
    throw std::invalid_argument(
        "GGML_MMQ_UNSUPPORTED: no compatible fused gate/up tile fits this activation shape");
  }
  return low;
}

at::Tensor linear_mmq_impl(
    const at::Tensor& quantized_weight,
    const at::Tensor* gate_weight,
    const at::Tensor& input,
    int64_t output_features,
    int64_t type_id,
    int64_t gate_type_id,
    ggml_glu_op glu_op,
    float glu_limit,
    int64_t workspace_budget_bytes,
    bool profile) {
  const bool fused = gate_weight != nullptr;
  TORCH_CHECK(input.is_cuda(), "ggml MMQ requires CUDA activations");
  TORCH_CHECK(quantized_weight.is_cuda(), "ggml MMQ requires CUDA GGUF weights");
  TORCH_CHECK(input.device() == quantized_weight.device(), "input and GGUF weights must use the same device");
  TORCH_CHECK(supported_type(static_cast<int>(type_id)), "unsupported GGUF quantization type");
  if (fused) {
    if (gate_type_id != type_id) {
      throw std::invalid_argument("GGML_MMQ_UNSUPPORTED: gate/up fusion requires identical quantization types");
    }
    TORCH_CHECK(gate_weight->is_cuda(), "ggml MMQ gate weights must be CUDA tensors");
    TORCH_CHECK(input.device() == gate_weight->device(), "input and gate GGUF weights must use the same device");
    TORCH_CHECK(gate_weight->is_contiguous(), "ggml MMQ requires contiguous packed gate weights");
    TORCH_CHECK(
        gate_weight->scalar_type() == quantized_weight.scalar_type(),
        "GGML_MMQ_UNSUPPORTED: packed gate/up storage dtypes differ");
  }
  TORCH_CHECK(input.dim() == 2, "ggml MMQ expects a two-dimensional activation matrix");
  TORCH_CHECK(input.is_contiguous(), "ggml MMQ requires contiguous activations");
  TORCH_CHECK(quantized_weight.is_contiguous(), "ggml MMQ requires contiguous packed weights");
  TORCH_CHECK(output_features > 0, "output feature count must be positive");
  TORCH_CHECK(workspace_budget_bytes >= 0, "workspace budget must not be negative");
  TORCH_CHECK(
      workspace_budget_bytes == kDefaultWorkspaceBudgetBytes || workspace_budget_bytes > kWorkspaceMargin,
      "an explicit workspace budget must be greater than 256 KiB");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16 ||
          input.scalar_type() == at::kFloat,
      "ggml MMQ accepts FP16, BF16, or FP32 activations");
  const int input_type_id = input.scalar_type() == at::kHalf
      ? GGML_TYPE_F16
      : input.scalar_type() == at::kBFloat16 ? GGML_TYPE_BF16 : GGML_TYPE_F32;

  const int64_t rows = input.size(0);
  const int64_t inner = input.size(1);
  const int64_t columns = output_features;
  TORCH_CHECK(inner > 0 && columns > 0, "matrix dimensions must be positive");
  if (fused) {
    const auto type = static_cast<ggml_type>(type_id);
    if (inner % ggml_blck_size(type) != 0) {
      throw std::invalid_argument("GGML_MMQ_UNSUPPORTED: gate/up K dimension is not aligned to the GGUF block size");
    }
    if (inner % ggml_blck_size(type) != 0) {
      throw std::invalid_argument("GGML_MMQ_UNSUPPORTED: gate/up K dimension is not aligned to the GGUF block size");
    }
    const uint64_t expected_bytes = static_cast<uint64_t>(ggml_row_size(type, inner)) *
        static_cast<uint64_t>(columns);
    const uint64_t up_bytes = static_cast<uint64_t>(quantized_weight.numel()) * quantized_weight.element_size();
    const uint64_t gate_bytes = static_cast<uint64_t>(gate_weight->numel()) * gate_weight->element_size();
    if (up_bytes != expected_bytes || gate_bytes != expected_bytes) {
      throw std::invalid_argument(
          "GGML_MMQ_UNSUPPORTED: packed gate/up byte lengths do not match the declared quantized matrix shape");
    }
    const bool supported_glu = glu_op == GGML_GLU_OP_SWIGLU || glu_op == GGML_GLU_OP_GEGLU ||
        glu_op == GGML_GLU_OP_SWIGLU_OAI || glu_op == GGML_GLU_OP_SWIGLU_CLAMP;
    if (!supported_glu || !std::isfinite(glu_limit) ||
        (glu_op == GGML_GLU_OP_SWIGLU_CLAMP ? glu_limit <= 0.0f : glu_limit != 0.0f)) {
      throw std::invalid_argument("GGML_MMQ_UNSUPPORTED: unsupported CUDA GLU operation or limit");
    }
  }
  std::unique_ptr<c10::cuda::CUDAGuard> device_guard;
  if (profile || fused) {
    device_guard = std::make_unique<c10::cuda::CUDAGuard>(input.device());
  }
  const bool direct_bf16_output = !fused && type_id == GGML_TYPE_Q8_0 &&
      input.scalar_type() == at::kBFloat16;
  const bool direct_output = input.scalar_type() == at::kFloat || direct_bf16_output;
  const int output_type_id = direct_bf16_output ? GGML_TYPE_BF16 : GGML_TYPE_F32;
  // The GGML MMQ bridge writes FP32. Only an FP32 result can target the final
  // tensor directly without changing GGML's accumulation/writeback contract.
  const size_t output_staging_element_size = direct_output ? 0 : sizeof(float);
  const int fusion_op_id = fused ? static_cast<int>(glu_op) : -1;
  const PerfKey perf_key{input.get_device(), static_cast<int>(type_id), static_cast<int>(input.scalar_type()),
                         fusion_op_id, rows, columns, inner};
  PerfCall perf_call(profile, perf_key, direct_output);

  if (rows == 0) {
    perf_call.complete();
    return at::empty({0, columns}, input.options());
  }
  TORCH_CHECK(
      inner <= std::numeric_limits<int>::max() && columns <= std::numeric_limits<int>::max(),
      "matrix dimensions exceed ggml CUDA MMQ limits");

  const int device = input.get_device();
  const auto caller_stream = c10::cuda::getCurrentCUDAStream(device);
  // GGML borrows these buffers and submits all work on the caller stream.
  // ComfyUI's weight contexts own pinning and stream fences. Their VBAR data
  // pointers need not belong to the PyTorch allocator.
  const auto plan_start = std::chrono::steady_clock::now();
  const int64_t chunk_capacity = choose_chunk_rows(
      device, static_cast<int>(type_id), rows, columns, inner, output_staging_element_size, workspace_budget_bytes,
      fused, fused ? static_cast<int>(glu_op) : -1, glu_limit);
  const auto plan_end = std::chrono::steady_clock::now();
  perf_call.add_plan(std::chrono::duration<double, std::milli>(plan_end - plan_start).count());
  if (chunk_capacity <= 0) {
    throw std::runtime_error(
        "GGML MMQ workspace budget cannot fit one row; increase COMFYUI_GGUF_WORKSPACE_BUDGET_MIB. "
        "The bridge does not reduce the budget or switch compute paths automatically.");
  }

  auto output = at::empty({rows, columns}, input.options());
  at::Tensor output_f32_staging;
  if (!direct_output) {
    // Reuse one full-capacity FP32 buffer for all row chunks. MMQ and copy_ are
    // enqueued on the same caller stream, so each copy completes before reuse.
    output_f32_staging = at::empty({chunk_capacity, columns}, input.options().dtype(at::kFloat));
    perf_call.add_output_staging_allocation();
  }

  int64_t row = 0;
  while (row < rows) {
    const int64_t remaining = rows - row;
    int64_t count = std::min<int64_t>(chunk_capacity, remaining);
    if (remaining < chunk_capacity) {
      const auto tail_plan_start = std::chrono::steady_clock::now();
      count = choose_chunk_rows(device, static_cast<int>(type_id), remaining, columns, inner,
                                output_staging_element_size, workspace_budget_bytes,
                                fused, fused ? static_cast<int>(glu_op) : -1, glu_limit);
      const auto tail_plan_end = std::chrono::steady_clock::now();
      perf_call.add_plan(std::chrono::duration<double, std::milli>(tail_plan_end - tail_plan_start).count());
      if (count <= 0) {
        throw std::runtime_error(
            "GGML MMQ workspace budget cannot fit the final row chunk; increase "
            "COMFYUI_GGUF_WORKSPACE_BUDGET_MIB. The bridge does not retry with a smaller budget.");
      }
      count = std::min<int64_t>(chunk_capacity, count);
    }
    perf_call.add_chunk();

    const auto caller_cuda_stream = caller_stream.stream() ? caller_stream.stream() : cudaStreamLegacy;
    PerfTimer input_timer(profile, perf_call.sample(), perf_key, PerfPhase::InputPrepare, caller_cuda_stream);
    auto input_chunk = input.narrow(0, row, count).contiguous();
    input_timer.finish(caller_cuda_stream);
    auto output_chunk = output.narrow(0, row, count);
    at::Tensor output_f32_chunk;
    void* output_data = nullptr;
    if (direct_output) {
      output_data = output_chunk.data_ptr();
    } else {
      output_f32_chunk = output_f32_staging.narrow(0, 0, count);
      output_data = output_f32_chunk.data_ptr();
    }

    std::array<char, 512> error{};
    void* compute_stream_ptr = nullptr;
    PerfTimer mmq_timer(profile, perf_call.sample(), perf_key, PerfPhase::ActivationQuantAndMmq, caller_cuda_stream);
    const int status = fused
        ? ggml_cuda_mmq_external_glu(
              device,
              quantized_weight.data_ptr(),
              gate_weight->data_ptr(),
              static_cast<int>(type_id),
              input_chunk.data_ptr(),
              input_type_id,
              static_cast<int>(glu_op),
              glu_limit,
              output_type_id,
              output_data,
              count,
              columns,
              inner,
              reinterpret_cast<void*>(caller_stream.stream()),
              &compute_stream_ptr,
              error.data(),
              error.size())
        : ggml_cuda_mmq_external(
              device,
              quantized_weight.data_ptr(),
              static_cast<int>(type_id),
              input_chunk.data_ptr(),
              input_type_id,
              output_type_id,
              output_data,
              count,
              columns,
              inner,
              reinterpret_cast<void*>(caller_stream.stream()),
              &compute_stream_ptr,
              error.data(),
              error.size());
    mmq_timer.finish(caller_cuda_stream);

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

    if (!direct_output) {
      PerfTimer output_timer(profile, perf_call.sample(), perf_key, PerfPhase::OutputCastCopy, caller_cuda_stream);
      output_chunk.copy_(output_f32_chunk);
      output_timer.finish(caller_cuda_stream);
    }
    row += count;
  }
  perf_call.complete();
  return output;
}

at::Tensor linear_mmq(
    const at::Tensor& quantized_weight,
    const at::Tensor& input,
    int64_t output_features,
    int64_t type_id,
    int64_t workspace_budget_bytes,
    bool profile) {
  return linear_mmq_impl(
      quantized_weight, nullptr, input, output_features, type_id, type_id,
      GGML_GLU_OP_SWIGLU, 0.0f, workspace_budget_bytes, profile);
}

at::Tensor linear_glu_mmq(
    const at::Tensor& up_weight,
    const at::Tensor& gate_weight,
    const at::Tensor& input,
    int64_t output_features,
    int64_t up_type_id,
    int64_t gate_type_id,
    ggml_glu_op glu_op,
    float glu_limit,
    int64_t workspace_budget_bytes,
    bool profile) {
  return linear_mmq_impl(
      up_weight, &gate_weight, input, output_features, up_type_id, gate_type_id,
      glu_op, glu_limit, workspace_budget_bytes, profile);
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
  info["default_workspace_budget_bytes"] = kDefaultWorkspaceBudgetBytes;
  info["workspace_budget_environment_variable"] = "COMFYUI_GGUF_WORKSPACE_BUDGET_MIB";
  info["fusion_api_version"] = 1;
  info["fusion_weight_types"] = pybind11::make_tuple("Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ4_XS");
  info["fusion_glu_ops"] = pybind11::make_tuple("GEGLU", "SWIGLU", "SWIGLU_OAI", "SWIGLU_CLAMP");
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
  pybind11::enum_<ggml_glu_op>(module, "GGMLGluOp")
      .value("GEGLU", GGML_GLU_OP_GEGLU)
      .value("SWIGLU", GGML_GLU_OP_SWIGLU)
      .value("SWIGLU_OAI", GGML_GLU_OP_SWIGLU_OAI)
      .value("SWIGLU_CLAMP", GGML_GLU_OP_SWIGLU_CLAMP);
  module.def(
      "linear_mmq",
      &linear_mmq,
      pybind11::arg("quantized_weight"),
      pybind11::arg("input"),
      pybind11::arg("output_features"),
      pybind11::arg("type_id"),
      pybind11::arg("workspace_budget_bytes") = kDefaultWorkspaceBudgetBytes,
      pybind11::arg("profile") = false);
  module.def(
      "linear_glu_mmq",
      &linear_glu_mmq,
      pybind11::arg("up_weight"),
      pybind11::arg("gate_weight"),
      pybind11::arg("input"),
      pybind11::arg("output_features"),
      pybind11::arg("up_type_id"),
      pybind11::arg("gate_type_id"),
      pybind11::arg("glu_op"),
      pybind11::arg("glu_limit") = 0.0f,
      pybind11::arg("workspace_budget_bytes") = kDefaultWorkspaceBudgetBytes,
      pybind11::arg("profile") = false);
  module.def("build_info", &build_info);
  module.def("collect_performance_stats", &collect_performance_stats);
  module.def("reset_performance_stats", &reset_performance_stats);
}
