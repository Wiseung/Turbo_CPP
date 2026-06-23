#include "turbo_cpp/lte_turbo.h"

#if TURBO_CPP_HAS_CUDA

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace turbo_cpp {
namespace {

namespace cg = cooperative_groups;

constexpr int kStates = 8;
constexpr float kNegInf = -1.0e30f;
constexpr int kStage2WindowSize = 128;
constexpr int kStage2ParallelShellBlockSize = 32;
constexpr int kStage2ParallelForwardWarmup = 32;
constexpr bool kStage2UseTile8ForwardBoundaryApply = false;
constexpr bool kStage2UseTile8ForwardBoundaryCompose = false;


#define TURBO_CUDA_CHECK(call)                                                     \
  do {                                                                             \
    cudaError_t err__ = (call);                                                    \
    if (err__ != cudaSuccess) {                                                    \
      throw std::runtime_error(cudaGetErrorString(err__));                         \
    }                                                                              \
  } while (0)

struct DeviceCodeBlockDescriptor {
  int k_r;
  int d_r;
  int filler_count;
};

struct Stage2WindowGeometry {
  int start;
  int local_k;
};

struct Stage2ScratchViews {
  float* alpha_in;
  float* beta_in;
  float* alpha_window;
  float* beta_window;
};

struct Stage2LaunchConfig {
  int window_size;
  int window_count;
  std::size_t shared_bytes;
};

struct Stage2ForwardBoundaryOperator {
  float transition[kStates * kStates];
};

__host__ __device__ int stage2_window_count_for_k(int k, int window_size) {
  return (k + window_size - 1) / window_size;
}

__host__ __device__ Stage2WindowGeometry stage2_window_geometry(int window_index,
                                                                int k,
                                                                int window_size) {
  const int start = window_index * window_size;
  const int end = (start + window_size < k) ? (start + window_size) : k;
  return Stage2WindowGeometry{start, end - start};
}

Stage2LaunchConfig make_stage2_launch_config(int k, int window_size) {
  const int window_count = stage2_window_count_for_k(k, window_size);
  const std::size_t shared_bytes =
      static_cast<std::size_t>((2 * window_count + 2 * (window_size + 1)) * kStates) *
      sizeof(float);
  return Stage2LaunchConfig{window_size, window_count, shared_bytes};
}

// Shared-memory footprint of the P2b stored-alpha kernel: full alpha trellis plus
// the windowed beta scratch. Used by the host to decide whether the fast path fits.
std::size_t stage2_stored_alpha_shared_bytes(int k, int window_size) {
  const int window_count = stage2_window_count_for_k(k, window_size);
  return static_cast<std::size_t>(((k + 1) + window_count + (window_size + 1)) * kStates) *
         sizeof(float);
}

__host__ __device__ int stage2_forward_boundary_operator_index(int next_state, int prev_state) {
  return next_state * kStates + prev_state;
}

__device__ float device_max_star(float x, float y) {
  if (!isfinite(x)) {
    return y;
  }
  if (!isfinite(y)) {
    return x;
  }
  const float hi = fmaxf(x, y);
  const float lo = fminf(x, y);
  return hi + log1pf(expf(lo - hi));
}

__device__ unsigned char device_feedback(unsigned char state, unsigned char input) {
  const unsigned char s1 = (state >> 1U) & 1U;
  const unsigned char s2 = (state >> 2U) & 1U;
  return static_cast<unsigned char>(input ^ s1 ^ s2);
}

__device__ unsigned char device_next_state(unsigned char state, unsigned char input) {
  const unsigned char s1 = (state >> 1U) & 1U;
  const unsigned char s2 = (state >> 2U) & 1U;
  const unsigned char feedback = static_cast<unsigned char>(input ^ s1 ^ s2);
  return static_cast<unsigned char>((feedback << 2U) | (s2 << 1U) | s1);
}

__device__ unsigned char device_parity(unsigned char state, unsigned char input) {
  const unsigned char s0 = state & 1U;
  const unsigned char s2 = (state >> 2U) & 1U;
  const unsigned char feedback = device_feedback(state, input);
  return static_cast<unsigned char>(feedback ^ s0 ^ s2);
}

__device__ void device_termination_inputs(unsigned char state, unsigned char* out) {
  for (int i = 0; i < 3; ++i) {
    const unsigned char s1 = (state >> 1U) & 1U;
    const unsigned char s2 = (state >> 2U) & 1U;
    const unsigned char input = static_cast<unsigned char>(s1 ^ s2);
    out[i] = input;
    state = device_next_state(state, input);
  }
}

__device__ void device_normalize(float* metrics) {
  float mx = metrics[0];
  for (int i = 1; i < kStates; ++i) {
    mx = fmaxf(mx, metrics[i]);
  }
  if (!isfinite(mx)) {
    return;
  }
  for (int i = 0; i < kStates; ++i) {
    if (isfinite(metrics[i])) {
      metrics[i] -= mx;
    }
  }
}

__device__ void device_build_tail_beta(float* beta_out,
                                       const float* tail_systematic_llr,
                                       const float* tail_parity_llr) {
  for (int s = 0; s < kStates; ++s) {
    beta_out[s] = kNegInf;
  }
  for (int s = 0; s < kStates; ++s) {
    float metric = 0.0f;
    unsigned char current = static_cast<unsigned char>(s);
    unsigned char inputs[3];
    device_termination_inputs(current, inputs);
    for (int step = 0; step < 3; ++step) {
      const unsigned char parity = device_parity(current, inputs[step]);
      current = device_next_state(current, inputs[step]);
      metric += 0.5f * ((inputs[step] ? 1.0f : -1.0f) * tail_systematic_llr[step]);
      metric += 0.5f * ((parity ? 1.0f : -1.0f) * tail_parity_llr[step]);
    }
    if (current == 0U) {
      beta_out[s] = metric;
    }
  }
  device_normalize(beta_out);
}

__global__ void interleave_kernel(const float* input, const int* perm, float* output, int k) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < k) {
    output[idx] = input[perm[idx]];
  }
}

__global__ void deinterleave_kernel(const float* input, const int* inverse_perm, float* output, int k) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < k) {
    output[idx] = input[inverse_perm[idx]];
  }
}

__global__ void posterior_kernel(const float* sys_llr,
                                 const float* ext1,
                                 const float* ext2,
                                 float* posterior,
                                 int k) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < k) {
    posterior[idx] = sys_llr[idx] + ext1[idx] + ext2[idx];
  }
}

__global__ void siso_exact_kernel(const float* sys_llr,
                                  const float* parity_llr,
                                  const float* apriori_llr,
                                  const float* tail_systematic_llr,
                                  const float* tail_parity_llr,
                                  float* extrinsic,
                                  DeviceCodeBlockDescriptor descriptor) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int k = descriptor.k_r;
  extern __shared__ float scratch[];
  float* alpha = scratch;
  float* beta = alpha + (k + 1) * kStates;

  for (int i = 0; i < (k + 1) * kStates; ++i) {
    alpha[i] = kNegInf;
    beta[i] = kNegInf;
  }
  alpha[0] = 0.0f;
  device_build_tail_beta(&beta[k * kStates], tail_systematic_llr, tail_parity_llr);

  for (int idx = 0; idx < k; ++idx) {
    float next_row[kStates];
    for (int s = 0; s < kStates; ++s) {
        next_row[s] = kNegInf;
    }
    for (int prev = 0; prev < kStates; ++prev) {
      const float alpha_prev = alpha[idx * kStates + prev];
      if (!isfinite(alpha_prev)) {
        continue;
      }
      for (unsigned char u = 0; u < 2; ++u) {
        const unsigned char parity = device_parity(static_cast<unsigned char>(prev), u);
        const unsigned char next = device_next_state(static_cast<unsigned char>(prev), u);
        const float gamma = 0.5f * ((u ? 1.0f : -1.0f) * (sys_llr[idx] + apriori_llr[idx]) +
                                    (parity ? 1.0f : -1.0f) * parity_llr[idx]);
        next_row[next] = device_max_star(next_row[next], alpha_prev + gamma);
      }
    }
    device_normalize(next_row);
    for (int s = 0; s < kStates; ++s) {
      alpha[(idx + 1) * kStates + s] = next_row[s];
    }
  }

  for (int idx = k - 1; idx >= 0; --idx) {
    float beta_row[kStates];
    for (int s = 0; s < kStates; ++s) {
        beta_row[s] = kNegInf;
    }
    for (int prev = 0; prev < kStates; ++prev) {
      for (unsigned char u = 0; u < 2; ++u) {
        const unsigned char parity = device_parity(static_cast<unsigned char>(prev), u);
        const unsigned char next = device_next_state(static_cast<unsigned char>(prev), u);
        const float next_beta = beta[(idx + 1) * kStates + next];
        if (!isfinite(next_beta)) {
          continue;
        }
        const float gamma = 0.5f * ((u ? 1.0f : -1.0f) * (sys_llr[idx] + apriori_llr[idx]) +
                                    (parity ? 1.0f : -1.0f) * parity_llr[idx]);
        beta_row[prev] = device_max_star(beta_row[prev], next_beta + gamma);
      }
    }
    device_normalize(beta_row);
    for (int s = 0; s < kStates; ++s) {
      beta[idx * kStates + s] = beta_row[s];
    }
  }

  for (int idx = 0; idx < k; ++idx) {
    float llr0 = kNegInf;
    float llr1 = kNegInf;
    for (int prev = 0; prev < kStates; ++prev) {
      const float alpha_prev = alpha[idx * kStates + prev];
      if (!isfinite(alpha_prev)) {
        continue;
      }
      for (unsigned char u = 0; u < 2; ++u) {
        const unsigned char parity = device_parity(static_cast<unsigned char>(prev), u);
        const unsigned char next = device_next_state(static_cast<unsigned char>(prev), u);
        const float gamma = 0.5f * ((u ? 1.0f : -1.0f) * (sys_llr[idx] + apriori_llr[idx]) +
                                    (parity ? 1.0f : -1.0f) * parity_llr[idx]);
        const float metric = alpha_prev + gamma + beta[(idx + 1) * kStates + next];
        if (u == 0U) {
          llr0 = device_max_star(llr0, metric);
        } else {
          llr1 = device_max_star(llr1, metric);
        }
      }
    }
    extrinsic[idx] = (llr1 - llr0) - sys_llr[idx] - apriori_llr[idx];
  }
}

__global__ void siso_exact_kernel_tile8_experimental(const float* sys_llr,
                                                     const float* parity_llr,
                                                     const float* apriori_llr,
                                                     const float* tail_systematic_llr,
                                                     const float* tail_parity_llr,
                                                     float* extrinsic,
                                                     DeviceCodeBlockDescriptor descriptor) {
  if (blockIdx.x != 0) {
    return;
  }

  const auto block = cg::this_thread_block();
  const auto tile = cg::tiled_partition<8>(block);
  if (tile.meta_group_rank() != 0) {
    return;
  }
  const int lane = static_cast<int>(tile.thread_rank());
  const int k = descriptor.k_r;

  extern __shared__ float scratch[];
  float* alpha = scratch;
  float* beta = alpha + (k + 1) * kStates;

  for (int i = lane; i < (k + 1) * kStates; i += kStates) {
    alpha[i] = kNegInf;
    beta[i] = kNegInf;
  }
  if (lane == 0) {
    alpha[0] = 0.0f;
    device_build_tail_beta(&beta[k * kStates], tail_systematic_llr, tail_parity_llr);
  }
  block.sync();

  for (int idx = 0; idx < k; ++idx) {
    float next_metric = kNegInf;
    for (int prev = 0; prev < kStates; ++prev) {
      const float alpha_prev = alpha[idx * kStates + prev];
      if (!isfinite(alpha_prev)) {
        continue;
      }
      for (unsigned char u = 0; u < 2; ++u) {
        const unsigned char next = device_next_state(static_cast<unsigned char>(prev), u);
        if (next != lane) {
          continue;
        }
        const unsigned char parity = device_parity(static_cast<unsigned char>(prev), u);
        const float gamma = 0.5f * ((u ? 1.0f : -1.0f) * (sys_llr[idx] + apriori_llr[idx]) +
                                    (parity ? 1.0f : -1.0f) * parity_llr[idx]);
        next_metric = device_max_star(next_metric, alpha_prev + gamma);
      }
    }
    alpha[(idx + 1) * kStates + lane] = next_metric;
    block.sync();
    if (lane == 0) {
      device_normalize(&alpha[(idx + 1) * kStates]);
    }
    block.sync();
  }

  for (int idx = k - 1; idx >= 0; --idx) {
    float beta_metric = kNegInf;
    for (unsigned char u = 0; u < 2; ++u) {
      const unsigned char next = device_next_state(static_cast<unsigned char>(lane), u);
      const float next_beta = beta[(idx + 1) * kStates + next];
      if (!isfinite(next_beta)) {
        continue;
      }
      const unsigned char parity = device_parity(static_cast<unsigned char>(lane), u);
      const float gamma = 0.5f * ((u ? 1.0f : -1.0f) * (sys_llr[idx] + apriori_llr[idx]) +
                                  (parity ? 1.0f : -1.0f) * parity_llr[idx]);
      beta_metric = device_max_star(beta_metric, next_beta + gamma);
    }
    beta[idx * kStates + lane] = beta_metric;
    block.sync();
    if (lane == 0) {
      device_normalize(&beta[idx * kStates]);
    }
    block.sync();
  }

  for (int idx = lane; idx < k; idx += kStates) {
    float llr0 = kNegInf;
    float llr1 = kNegInf;
    for (int prev = 0; prev < kStates; ++prev) {
      const float alpha_prev = alpha[idx * kStates + prev];
      if (!isfinite(alpha_prev)) {
        continue;
      }
      for (unsigned char u = 0; u < 2; ++u) {
        const unsigned char parity = device_parity(static_cast<unsigned char>(prev), u);
        const unsigned char next = device_next_state(static_cast<unsigned char>(prev), u);
        const float gamma = 0.5f * ((u ? 1.0f : -1.0f) * (sys_llr[idx] + apriori_llr[idx]) +
                                    (parity ? 1.0f : -1.0f) * parity_llr[idx]);
        const float metric = alpha_prev + gamma + beta[(idx + 1) * kStates + next];
        if (u == 0U) {
          llr0 = device_max_star(llr0, metric);
        } else {
          llr1 = device_max_star(llr1, metric);
        }
      }
    }
    extrinsic[idx] = (llr1 - llr0) - sys_llr[idx] - apriori_llr[idx];
  }
}

__device__ void device_copy_state_row(float* dst, const float* src) {
  for (int s = 0; s < kStates; ++s) {
    dst[s] = src[s];
  }
}

template <typename TileGroup>
__device__ void device_fill_state_row_tile8(float* dst, float value, TileGroup tile) {
  const int lane = static_cast<int>(tile.thread_rank());
  if (lane < kStates) {
    dst[lane] = value;
  }
  tile.sync();
}

template <typename TileGroup>
__device__ void device_stage2_window_forward_alpha_tile8(const float* sys_llr,
                                                         const float* parity_llr,
                                                         const float* apriori_llr,
                                                         int start,
                                                         int local_k,
                                                         const float* alpha_init,
                                                         float* alpha_window,
                                                         TileGroup tile) {
  const int lane = static_cast<int>(tile.thread_rank());
  if (lane < kStates) {
    alpha_window[lane] = alpha_init[lane];
  }
  tile.sync();

  for (int idx = 0; idx < local_k; ++idx) {
    const int global_idx = start + idx;
    float next_metric = kNegInf;
    for (int prev = 0; prev < kStates; ++prev) {
      const float alpha_prev = alpha_window[idx * kStates + prev];
      if (!isfinite(alpha_prev)) {
        continue;
      }
      for (unsigned char u = 0; u < 2; ++u) {
        const unsigned char next = device_next_state(static_cast<unsigned char>(prev), u);
        if (next != lane) {
          continue;
        }
        const unsigned char parity = device_parity(static_cast<unsigned char>(prev), u);
        const float gamma = 0.5f *
                            ((u ? 1.0f : -1.0f) * (sys_llr[global_idx] + apriori_llr[global_idx]) +
                             (parity ? 1.0f : -1.0f) * parity_llr[global_idx]);
        next_metric = device_max_star(next_metric, alpha_prev + gamma);
      }
    }
    if (lane < kStates) {
      alpha_window[(idx + 1) * kStates + lane] = next_metric;
    }
    tile.sync();
    if (lane == 0) {
      device_normalize(&alpha_window[(idx + 1) * kStates]);
    }
    tile.sync();
  }
}

template <typename TileGroup>
__device__ void device_stage2_backward_window_beta_tile8(const float* sys_llr,
                                                         const float* parity_llr,
                                                         const float* apriori_llr,
                                                         int start,
                                                         int local_k,
                                                         const float* beta_init,
                                                         float* beta_window,
                                                         TileGroup tile) {
  const int lane = static_cast<int>(tile.thread_rank());
  if (lane < kStates) {
    beta_window[local_k * kStates + lane] = beta_init[lane];
  }
  tile.sync();

  for (int idx = local_k - 1; idx >= 0; --idx) {
    const int global_idx = start + idx;
    float beta_metric = kNegInf;
    for (unsigned char u = 0; u < 2; ++u) {
      const unsigned char next = device_next_state(static_cast<unsigned char>(lane), u);
      const float next_beta = beta_window[(idx + 1) * kStates + next];
      if (!isfinite(next_beta)) {
        continue;
      }
      const unsigned char parity = device_parity(static_cast<unsigned char>(lane), u);
      const float gamma = 0.5f *
                          ((u ? 1.0f : -1.0f) * (sys_llr[global_idx] + apriori_llr[global_idx]) +
                           (parity ? 1.0f : -1.0f) * parity_llr[global_idx]);
      beta_metric = device_max_star(beta_metric, next_beta + gamma);
    }
    if (lane < kStates) {
      beta_window[idx * kStates + lane] = beta_metric;
    }
    tile.sync();
    if (lane == 0) {
      device_normalize(&beta_window[idx * kStates]);
    }
    tile.sync();
  }
}

template <typename TileGroup>
__device__ void device_stage2_window_llr_tile8(const float* sys_llr,
                                               const float* parity_llr,
                                               const float* apriori_llr,
                                               float* extrinsic,
                                               int start,
                                               int local_k,
                                               const float* alpha_window,
                                               const float* beta_window,
                                               TileGroup tile) {
  const int lane = static_cast<int>(tile.thread_rank());
  for (int idx = lane; idx < local_k; idx += kStates) {
    const int global_idx = start + idx;
    float llr0 = kNegInf;
    float llr1 = kNegInf;
    for (int prev = 0; prev < kStates; ++prev) {
      const float alpha_prev = alpha_window[idx * kStates + prev];
      if (!isfinite(alpha_prev)) {
        continue;
      }
      for (unsigned char u = 0; u < 2; ++u) {
        const unsigned char parity = device_parity(static_cast<unsigned char>(prev), u);
        const unsigned char next = device_next_state(static_cast<unsigned char>(prev), u);
        const float gamma = 0.5f *
                            ((u ? 1.0f : -1.0f) * (sys_llr[global_idx] + apriori_llr[global_idx]) +
                             (parity ? 1.0f : -1.0f) * parity_llr[global_idx]);
        const float metric = alpha_prev + gamma + beta_window[(idx + 1) * kStates + next];
        if (u == 0U) {
          llr0 = device_max_star(llr0, metric);
        } else {
          llr1 = device_max_star(llr1, metric);
        }
      }
    }
    extrinsic[global_idx] = (llr1 - llr0) - sys_llr[global_idx] - apriori_llr[global_idx];
  }
}

__device__ void device_stage2_init_forward_boundary_operator(Stage2ForwardBoundaryOperator* op) {
  for (int i = 0; i < kStates * kStates; ++i) {
    op->transition[i] = kNegInf;
  }
}

__device__ void device_stage2_forward_boundary_operator_compose_step_reference(
    Stage2ForwardBoundaryOperator* op,
    float sys_plus_apriori,
    float parity_llr_value) {
  float next_transition[kStates * kStates];
  for (int i = 0; i < kStates * kStates; ++i) {
    next_transition[i] = kNegInf;
  }

  for (int next = 0; next < kStates; ++next) {
    for (int origin = 0; origin < kStates; ++origin) {
      float metric = kNegInf;
      for (int prev = 0; prev < kStates; ++prev) {
        const float incoming = op->transition[stage2_forward_boundary_operator_index(prev, origin)];
        if (!isfinite(incoming)) {
          continue;
        }
        for (unsigned char u = 0; u < 2; ++u) {
          const unsigned char candidate_next =
              device_next_state(static_cast<unsigned char>(prev), u);
          if (candidate_next != next) {
            continue;
          }
          const unsigned char parity = device_parity(static_cast<unsigned char>(prev), u);
          const float gamma = 0.5f * ((u ? 1.0f : -1.0f) * sys_plus_apriori +
                                      (parity ? 1.0f : -1.0f) * parity_llr_value);
          metric = device_max_star(metric, incoming + gamma);
        }
      }
      next_transition[stage2_forward_boundary_operator_index(next, origin)] = metric;
    }
  }

  for (int origin = 0; origin < kStates; ++origin) {
    float column[kStates];
    for (int next = 0; next < kStates; ++next) {
      column[next] = next_transition[stage2_forward_boundary_operator_index(next, origin)];
    }
    device_normalize(column);
    for (int next = 0; next < kStates; ++next) {
      next_transition[stage2_forward_boundary_operator_index(next, origin)] = column[next];
    }
  }

  for (int i = 0; i < kStates * kStates; ++i) {
    op->transition[i] = next_transition[i];
  }
}

template <typename TileGroup>
__device__ void device_stage2_forward_boundary_operator_compose_step_tile8(
    Stage2ForwardBoundaryOperator* op,
    float sys_plus_apriori,
    float parity_llr_value,
    TileGroup tile) {
  const int lane = static_cast<int>(tile.thread_rank());
  float next_transition_local[kStates];
  for (int origin = 0; origin < kStates; ++origin) {
    float metric = kNegInf;
    for (int prev = 0; prev < kStates; ++prev) {
      const float incoming = op->transition[stage2_forward_boundary_operator_index(prev, origin)];
      if (!isfinite(incoming)) {
        continue;
      }
      for (unsigned char u = 0; u < 2; ++u) {
        const unsigned char candidate_next =
            device_next_state(static_cast<unsigned char>(prev), u);
        if (candidate_next != lane) {
          continue;
        }
        const unsigned char parity = device_parity(static_cast<unsigned char>(prev), u);
        const float gamma = 0.5f * ((u ? 1.0f : -1.0f) * sys_plus_apriori +
                                    (parity ? 1.0f : -1.0f) * parity_llr_value);
        metric = device_max_star(metric, incoming + gamma);
      }
    }
    next_transition_local[origin] = metric;
  }

  for (int origin = 0; origin < kStates; ++origin) {
    op->transition[stage2_forward_boundary_operator_index(lane, origin)] =
        next_transition_local[origin];
  }
  tile.sync();

  if (lane == 0) {
    for (int origin = 0; origin < kStates; ++origin) {
      float column[kStates];
      for (int next = 0; next < kStates; ++next) {
        column[next] = op->transition[stage2_forward_boundary_operator_index(next, origin)];
      }
      device_normalize(column);
      for (int next = 0; next < kStates; ++next) {
        op->transition[stage2_forward_boundary_operator_index(next, origin)] = column[next];
      }
    }
  }
  tile.sync();
}

template <typename TileGroup>
__device__ void device_stage2_forward_boundary_operator_compose_step_dispatch(
    Stage2ForwardBoundaryOperator* op,
    float sys_plus_apriori,
    float parity_llr_value,
    TileGroup tile) {
  if constexpr (kStage2UseTile8ForwardBoundaryCompose) {
    device_stage2_forward_boundary_operator_compose_step_tile8(
        op, sys_plus_apriori, parity_llr_value, tile);
  } else {
    if (tile.thread_rank() == 0) {
      device_stage2_forward_boundary_operator_compose_step_reference(
          op, sys_plus_apriori, parity_llr_value);
    }
    tile.sync();
  }
}

template <typename TileGroup>
__device__ void device_stage2_build_forward_boundary_operator(
    const float* sys_llr,
    const float* parity_llr,
    const float* apriori_llr,
    int start,
    int local_k,
    Stage2ForwardBoundaryOperator* op,
    TileGroup tile) {
  const int lane = static_cast<int>(tile.thread_rank());
  if (lane == 0) {
    device_stage2_init_forward_boundary_operator(op);
  }
  tile.sync();
  op->transition[stage2_forward_boundary_operator_index(lane, lane)] = 0.0f;
  tile.sync();
  for (int idx = 0; idx < local_k; ++idx) {
    const int global_idx = start + idx;
    device_stage2_forward_boundary_operator_compose_step_dispatch(
        op,
        sys_llr[global_idx] + apriori_llr[global_idx],
        parity_llr[global_idx],
        tile);
  }
}

__device__ void device_stage2_apply_forward_boundary_operator_reference(
    const Stage2ForwardBoundaryOperator* op,
    const float* alpha_in,
    float* alpha_out) {
  for (int next = 0; next < kStates; ++next) {
    float metric = kNegInf;
    for (int origin = 0; origin < kStates; ++origin) {
      const float alpha_origin = alpha_in[origin];
      const float transition = op->transition[stage2_forward_boundary_operator_index(next, origin)];
      if (!isfinite(alpha_origin) || !isfinite(transition)) {
        continue;
      }
      metric = device_max_star(metric, alpha_origin + transition);
    }
    alpha_out[next] = metric;
  }
  device_normalize(alpha_out);
}

template <typename TileGroup>
__device__ void device_stage2_apply_forward_boundary_operator_tile8(
    const Stage2ForwardBoundaryOperator* op,
    const float* alpha_in,
    float* alpha_out,
    TileGroup tile) {
  const int next = static_cast<int>(tile.thread_rank());
  float metric = kNegInf;
  for (int origin = 0; origin < kStates; ++origin) {
    const float alpha_origin = alpha_in[origin];
    const float transition = op->transition[stage2_forward_boundary_operator_index(next, origin)];
    if (!isfinite(alpha_origin) || !isfinite(transition)) {
      continue;
    }
    metric = device_max_star(metric, alpha_origin + transition);
  }
  alpha_out[next] = metric;
  tile.sync();
  if (next == 0) {
    device_normalize(alpha_out);
  }
  tile.sync();
}

template <typename TileGroup>
__device__ void device_stage2_apply_forward_boundary_operator_dispatch(
    const Stage2ForwardBoundaryOperator* op,
    const float* alpha_in,
    float* alpha_out,
    TileGroup tile) {
  if constexpr (kStage2UseTile8ForwardBoundaryApply) {
    device_stage2_apply_forward_boundary_operator_tile8(op, alpha_in, alpha_out, tile);
  } else {
    if (tile.thread_rank() == 0) {
      device_stage2_apply_forward_boundary_operator_reference(op, alpha_in, alpha_out);
    }
    tile.sync();
  }
}

__device__ void device_stage2_window_forward_alpha(const float* sys_llr,
                                                   const float* parity_llr,
                                                   const float* apriori_llr,
                                                   int start,
                                                   int local_k,
                                                   const float* alpha_init,
                                                   float* alpha_window) {
  device_copy_state_row(alpha_window, alpha_init);
  for (int idx = 0; idx < local_k; ++idx) {
    const int global_idx = start + idx;
    float next_row[kStates];
    for (int s = 0; s < kStates; ++s) {
      next_row[s] = kNegInf;
    }
    for (int prev = 0; prev < kStates; ++prev) {
      const float alpha_prev = alpha_window[idx * kStates + prev];
      if (!isfinite(alpha_prev)) {
        continue;
      }
      for (unsigned char u = 0; u < 2; ++u) {
        const unsigned char parity = device_parity(static_cast<unsigned char>(prev), u);
        const unsigned char next = device_next_state(static_cast<unsigned char>(prev), u);
        const float gamma = 0.5f * ((u ? 1.0f : -1.0f) * (sys_llr[global_idx] + apriori_llr[global_idx]) +
                                    (parity ? 1.0f : -1.0f) * parity_llr[global_idx]);
        next_row[next] = device_max_star(next_row[next], alpha_prev + gamma);
      }
    }
    device_normalize(next_row);
    for (int s = 0; s < kStates; ++s) {
      alpha_window[(idx + 1) * kStates + s] = next_row[s];
    }
  }
}

__device__ void device_stage2_forward_boundary_scan_reference(const float* sys_llr,
                                                              const float* parity_llr,
                                                              const float* apriori_llr,
                                                              int k,
                                                              int window_size,
                                                              int window_count,
                                                              float* alpha_in,
                                                              float* alpha_window) {
  for (int w = 0; w < window_count; ++w) {
    const Stage2WindowGeometry geometry = stage2_window_geometry(w, k, window_size);
    device_stage2_window_forward_alpha(sys_llr, parity_llr, apriori_llr,
                                       geometry.start, geometry.local_k,
                                       &alpha_in[w * kStates], alpha_window);
    if (w + 1 < window_count) {
      device_copy_state_row(&alpha_in[(w + 1) * kStates],
                            &alpha_window[geometry.local_k * kStates]);
    }
  }
}

template <typename TileGroup>
__device__ void device_stage2_forward_boundary_scan_tile8(const float* sys_llr,
                                                          const float* parity_llr,
                                                          const float* apriori_llr,
                                                          int k,
                                                          int window_size,
                                                          int window_count,
                                                          float* alpha_in,
                                                          Stage2ForwardBoundaryOperator* op,
                                                          TileGroup tile) {
  const int lane = static_cast<int>(tile.thread_rank());
  for (int w = 0; w < window_count; ++w) {
    const Stage2WindowGeometry geometry = stage2_window_geometry(w, k, window_size);
    device_stage2_build_forward_boundary_operator(
        sys_llr, parity_llr, apriori_llr, geometry.start, geometry.local_k, op, tile);
    if (w + 1 < window_count) {
      device_stage2_apply_forward_boundary_operator_dispatch(
          op,
          &alpha_in[w * kStates],
          &alpha_in[(w + 1) * kStates],
          tile);
    }
    tile.sync();
  }
}

__device__ void device_stage2_backward_boundary_scan_window(const float* sys_llr,
                                                            const float* parity_llr,
                                                            const float* apriori_llr,
                                                            int start,
                                                            int local_k,
                                                            const float* beta_init,
                                                            float* beta_window,
                                                            float* previous_boundary) {
  device_copy_state_row(&beta_window[local_k * kStates], beta_init);
  for (int idx = local_k - 1; idx >= 0; --idx) {
    const int global_idx = start + idx;
    float beta_row[kStates];
    for (int s = 0; s < kStates; ++s) {
      beta_row[s] = kNegInf;
    }
    for (int prev = 0; prev < kStates; ++prev) {
      for (unsigned char u = 0; u < 2; ++u) {
        const unsigned char parity = device_parity(static_cast<unsigned char>(prev), u);
        const unsigned char next = device_next_state(static_cast<unsigned char>(prev), u);
        const float next_beta = beta_window[(idx + 1) * kStates + next];
        if (!isfinite(next_beta)) {
          continue;
        }
        const float gamma = 0.5f * ((u ? 1.0f : -1.0f) * (sys_llr[global_idx] + apriori_llr[global_idx]) +
                                    (parity ? 1.0f : -1.0f) * parity_llr[global_idx]);
        beta_row[prev] = device_max_star(beta_row[prev], next_beta + gamma);
      }
    }
    device_normalize(beta_row);
    for (int s = 0; s < kStates; ++s) {
      beta_window[idx * kStates + s] = beta_row[s];
    }
  }
  if (previous_boundary != nullptr) {
    device_copy_state_row(previous_boundary, beta_window);
  }
}

__device__ void device_stage2_apply_window_operator(const float* sys_llr,
                                                    const float* parity_llr,
                                                    const float* apriori_llr,
                                                    float* extrinsic,
                                                    int start,
                                                    int local_k,
                                                    const float* alpha_init,
                                                    const float* beta_window,
                                                    float* alpha_window) {
  device_stage2_window_forward_alpha(sys_llr, parity_llr, apriori_llr,
                                     start, local_k, alpha_init, alpha_window);
  for (int idx = 0; idx < local_k; ++idx) {
    const int global_idx = start + idx;
    float llr0 = kNegInf;
    float llr1 = kNegInf;
    for (int prev = 0; prev < kStates; ++prev) {
      const float alpha_prev = alpha_window[idx * kStates + prev];
      if (!isfinite(alpha_prev)) {
        continue;
      }
      for (unsigned char u = 0; u < 2; ++u) {
        const unsigned char parity = device_parity(static_cast<unsigned char>(prev), u);
        const unsigned char next = device_next_state(static_cast<unsigned char>(prev), u);
        const float gamma = 0.5f * ((u ? 1.0f : -1.0f) * (sys_llr[global_idx] + apriori_llr[global_idx]) +
                                    (parity ? 1.0f : -1.0f) * parity_llr[global_idx]);
        const float metric = alpha_prev + gamma + beta_window[(idx + 1) * kStates + next];
        if (u == 0U) {
          llr0 = device_max_star(llr0, metric);
        } else {
          llr1 = device_max_star(llr1, metric);
        }
      }
    }
    extrinsic[global_idx] = (llr1 - llr0) - sys_llr[global_idx] - apriori_llr[global_idx];
  }
}

// P2b stored-alpha path: run the forward recursion once over the whole block and
// retain every alpha row in shared memory, so the LLR stage can read alpha back
// instead of recomputing a full forward pass per window. Mathematically identical
// to the windowed path (same recursion, same per-step normalize); only the number
// of forward passes changes (~4 -> ~3 single-threaded passes).
__device__ void device_stage2_full_forward_alpha(const float* sys_llr,
                                                 const float* parity_llr,
                                                 const float* apriori_llr,
                                                 int k,
                                                 float* alpha_full) {
  // alpha_full[0..kStates) is expected pre-initialized (alpha_full[0]=0, rest kNegInf).
  for (int idx = 0; idx < k; ++idx) {
    float next_row[kStates];
    for (int s = 0; s < kStates; ++s) {
      next_row[s] = kNegInf;
    }
    for (int prev = 0; prev < kStates; ++prev) {
      const float alpha_prev = alpha_full[idx * kStates + prev];
      if (!isfinite(alpha_prev)) {
        continue;
      }
      for (unsigned char u = 0; u < 2; ++u) {
        const unsigned char parity = device_parity(static_cast<unsigned char>(prev), u);
        const unsigned char next = device_next_state(static_cast<unsigned char>(prev), u);
        const float gamma = 0.5f * ((u ? 1.0f : -1.0f) * (sys_llr[idx] + apriori_llr[idx]) +
                                    (parity ? 1.0f : -1.0f) * parity_llr[idx]);
        next_row[next] = device_max_star(next_row[next], alpha_prev + gamma);
      }
    }
    device_normalize(next_row);
    for (int s = 0; s < kStates; ++s) {
      alpha_full[(idx + 1) * kStates + s] = next_row[s];
    }
  }
}

__device__ void device_stage2_window_llr_stored_alpha(const float* sys_llr,
                                                      const float* parity_llr,
                                                      const float* apriori_llr,
                                                      float* extrinsic,
                                                      int start,
                                                      int local_k,
                                                      const float* alpha_full,
                                                      const float* beta_window) {
  for (int idx = 0; idx < local_k; ++idx) {
    const int global_idx = start + idx;
    float llr0 = kNegInf;
    float llr1 = kNegInf;
    for (int prev = 0; prev < kStates; ++prev) {
      const float alpha_prev = alpha_full[global_idx * kStates + prev];
      if (!isfinite(alpha_prev)) {
        continue;
      }
      for (unsigned char u = 0; u < 2; ++u) {
        const unsigned char parity = device_parity(static_cast<unsigned char>(prev), u);
        const unsigned char next = device_next_state(static_cast<unsigned char>(prev), u);
        const float gamma = 0.5f * ((u ? 1.0f : -1.0f) * (sys_llr[global_idx] + apriori_llr[global_idx]) +
                                    (parity ? 1.0f : -1.0f) * parity_llr[global_idx]);
        const float metric = alpha_prev + gamma + beta_window[(idx + 1) * kStates + next];
        if (u == 0U) {
          llr0 = device_max_star(llr0, metric);
        } else {
          llr1 = device_max_star(llr1, metric);
        }
      }
    }
    extrinsic[global_idx] = (llr1 - llr0) - sys_llr[global_idx] - apriori_llr[global_idx];
  }
}

__global__ void siso_exact_windowed_kernel(const float* sys_llr,
                                           const float* parity_llr,
                                           const float* apriori_llr,
                                           const float* tail_systematic_llr,
                                           const float* tail_parity_llr,
                                           float* extrinsic,
                                           DeviceCodeBlockDescriptor descriptor,
                                           int window_size) {
  if (blockIdx.x != 0) {
    return;
  }
  const auto block = cg::this_thread_block();
  const auto tile = cg::tiled_partition<kStates>(block);
  if (tile.meta_group_rank() != 0) {
    return;
  }
  const int lane = static_cast<int>(tile.thread_rank());
  const int k = descriptor.k_r;
  const int window_count = stage2_window_count_for_k(k, window_size);

  extern __shared__ float scratch[];
  Stage2ScratchViews views;
  views.alpha_in = scratch;
  views.beta_in = views.alpha_in + window_count * kStates;
  views.alpha_window = views.beta_in + window_count * kStates;
  views.beta_window = views.alpha_window + (window_size + 1) * kStates;
  auto* forward_op = reinterpret_cast<Stage2ForwardBoundaryOperator*>(views.alpha_window);

  for (int i = lane; i < window_count * kStates; i += kStates) {
    views.alpha_in[i] = kNegInf;
    views.beta_in[i] = kNegInf;
  }
  if (lane == 0) {
    views.alpha_in[0] = 0.0f;
    device_build_tail_beta(&views.beta_in[(window_count - 1) * kStates],
                           tail_systematic_llr, tail_parity_llr);
  }
  block.sync();
  if (window_count > 1) {
    if constexpr (kStage2UseTile8ForwardBoundaryApply || kStage2UseTile8ForwardBoundaryCompose) {
      device_stage2_forward_boundary_scan_tile8(sys_llr, parity_llr, apriori_llr,
                                                k, window_size, window_count,
                                                views.alpha_in, forward_op, tile);
    } else {
      if (lane == 0) {
        device_stage2_forward_boundary_scan_reference(sys_llr, parity_llr, apriori_llr,
                                                      k, window_size, window_count,
                                                      views.alpha_in, views.alpha_window);
      }
      block.sync();
    }
  }
  block.sync();

  if (lane == 0) {
    for (int w = window_count - 1; w >= 0; --w) {
      const Stage2WindowGeometry geometry = stage2_window_geometry(w, k, window_size);
      device_stage2_backward_boundary_scan_window(sys_llr, parity_llr, apriori_llr,
                                                  geometry.start, geometry.local_k,
                                                  &views.beta_in[w * kStates], views.beta_window,
                                                  (w > 0) ? &views.beta_in[(w - 1) * kStates] : nullptr);
      device_stage2_apply_window_operator(sys_llr, parity_llr, apriori_llr, extrinsic,
                                          geometry.start, geometry.local_k,
                                          &views.alpha_in[w * kStates], views.beta_window,
                                          views.alpha_window);
    }
  }
}

// P2b fast path: identical math to siso_exact_windowed_kernel, but stores the full
// alpha trellis once (forward pass) and reuses it in the LLR stage, dropping the
// per-window alpha recomputation. Requires (k+1)*kStates floats of shared alpha, so
// the host only launches this when shared memory fits; otherwise it falls back to
// the windowed kernel above.
__global__ void siso_exact_stored_alpha_kernel(const float* sys_llr,
                                               const float* parity_llr,
                                               const float* apriori_llr,
                                               const float* tail_systematic_llr,
                                               const float* tail_parity_llr,
                                               float* extrinsic,
                                               DeviceCodeBlockDescriptor descriptor,
                                               int window_size) {
  if (blockIdx.x != 0) {
    return;
  }
  const auto block = cg::this_thread_block();
  const auto tile = cg::tiled_partition<kStates>(block);
  if (tile.meta_group_rank() != 0) {
    return;
  }
  const int lane = static_cast<int>(tile.thread_rank());
  const int k = descriptor.k_r;
  const int window_count = stage2_window_count_for_k(k, window_size);

  extern __shared__ float scratch[];
  float* alpha_full = scratch;                          // (k + 1) * kStates
  float* beta_in = alpha_full + (k + 1) * kStates;      // window_count * kStates
  float* beta_window = beta_in + window_count * kStates;  // (window_size + 1) * kStates

  for (int i = lane; i < (k + 1) * kStates; i += kStates) {
    alpha_full[i] = kNegInf;
  }
  for (int i = lane; i < window_count * kStates; i += kStates) {
    beta_in[i] = kNegInf;
  }
  if (lane == 0) {
    alpha_full[0] = 0.0f;
    device_build_tail_beta(&beta_in[(window_count - 1) * kStates],
                           tail_systematic_llr, tail_parity_llr);
  }
  block.sync();

  if (lane == 0) {
    device_stage2_full_forward_alpha(sys_llr, parity_llr, apriori_llr, k, alpha_full);
    for (int w = window_count - 1; w >= 0; --w) {
      const Stage2WindowGeometry geometry = stage2_window_geometry(w, k, window_size);
      device_stage2_backward_boundary_scan_window(sys_llr, parity_llr, apriori_llr,
                                                  geometry.start, geometry.local_k,
                                                  &beta_in[w * kStates], beta_window,
                                                  (w > 0) ? &beta_in[(w - 1) * kStates] : nullptr);
      device_stage2_window_llr_stored_alpha(sys_llr, parity_llr, apriori_llr, extrinsic,
                                            geometry.start, geometry.local_k,
                                            alpha_full, beta_window);
    }
  }
}

// P-A2a experimental shell: one warp-sized block per window. This is intentionally
// not a correct decoder path; it exists only to validate whether the large-K
// fallback benefits from exposing per-window parallel launch geometry.
__global__ void siso_parallel_window_shell_kernel(const float* sys_llr,
                                                  const float* parity_llr,
                                                  const float* apriori_llr,
                                                  float* extrinsic,
                                                  DeviceCodeBlockDescriptor descriptor,
                                                  int window_size) {
  const int k = descriptor.k_r;
  const int window_count = stage2_window_count_for_k(k, window_size);
  const int window_index = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(threadIdx.x & 31);
  if (window_index >= window_count || lane >= kStates) {
    return;
  }

  const Stage2WindowGeometry geometry =
      stage2_window_geometry(window_index, k, window_size);
  float state_metric = 0.0f;
  for (int local_idx = 0; local_idx < geometry.local_k; ++local_idx) {
    const int global_idx = geometry.start + local_idx;
    const float sys = sys_llr[global_idx];
    const float par = parity_llr[global_idx];
    const float apr = apriori_llr[global_idx];
    state_metric += sys * 0.125f + par * 0.0625f + apr * 0.03125f +
                    static_cast<float>((lane + local_idx) & 1);
    state_metric = fmaf(state_metric, 0.9995f, -0.0001f * static_cast<float>(lane));
  }

  for (int local_idx = lane; local_idx < geometry.local_k; local_idx += kStates) {
    const int global_idx = geometry.start + local_idx;
    extrinsic[global_idx] = state_metric;
  }
}

__global__ void siso_parallel_window_forward_experimental_kernel(
    const float* sys_llr,
    const float* parity_llr,
    const float* apriori_llr,
    float* forward_sink,
    DeviceCodeBlockDescriptor descriptor,
    int window_size,
    int warmup_length) {
  const auto block = cg::this_thread_block();
  const auto tile = cg::tiled_partition<kStates>(block);
  if (tile.meta_group_rank() != 0) {
    return;
  }

  const int k = descriptor.k_r;
  const int window_count = stage2_window_count_for_k(k, window_size);
  const int window_index = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(tile.thread_rank());
  if (window_index >= window_count) {
    return;
  }

  const Stage2WindowGeometry geometry =
      stage2_window_geometry(window_index, k, window_size);
  const int warmup_start =
      geometry.start > warmup_length ? (geometry.start - warmup_length) : 0;
  const int warmup_k = geometry.start - warmup_start;

  extern __shared__ float scratch[];
  float* alpha_init = scratch;
  float* alpha_window = alpha_init + kStates;

  if (warmup_start == 0) {
    device_fill_state_row_tile8(alpha_init, kNegInf, tile);
    if (lane == 0) {
      alpha_init[0] = 0.0f;
    }
    tile.sync();
  } else {
    device_fill_state_row_tile8(alpha_init, 0.0f, tile);
  }

  if (warmup_k > 0) {
    device_stage2_window_forward_alpha_tile8(
        sys_llr, parity_llr, apriori_llr, warmup_start, warmup_k,
        alpha_init, alpha_window, tile);
    if (lane < kStates) {
      alpha_init[lane] = alpha_window[warmup_k * kStates + lane];
    }
    tile.sync();
  }

  device_stage2_window_forward_alpha_tile8(
      sys_llr, parity_llr, apriori_llr, geometry.start, geometry.local_k,
      alpha_init, alpha_window, tile);

  for (int local_idx = lane; local_idx < geometry.local_k; local_idx += kStates) {
    const int global_idx = geometry.start + local_idx;
    forward_sink[global_idx] = alpha_window[(local_idx + 1) * kStates + 0];
  }
}

__global__ void siso_parallel_window_forward_backward_experimental_kernel(
    const float* sys_llr,
    const float* parity_llr,
    const float* apriori_llr,
    const float* tail_systematic_llr,
    const float* tail_parity_llr,
    float* sink,
    DeviceCodeBlockDescriptor descriptor,
    int window_size,
    int warmup_length) {
  const auto block = cg::this_thread_block();
  const auto tile = cg::tiled_partition<kStates>(block);
  if (tile.meta_group_rank() != 0) {
    return;
  }

  const int k = descriptor.k_r;
  const int window_count = stage2_window_count_for_k(k, window_size);
  const int window_index = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(tile.thread_rank());
  if (window_index >= window_count) {
    return;
  }

  const Stage2WindowGeometry geometry =
      stage2_window_geometry(window_index, k, window_size);
  const int warmup_alpha_start =
      geometry.start > warmup_length ? (geometry.start - warmup_length) : 0;
  const int warmup_alpha_k = geometry.start - warmup_alpha_start;
  const int beta_start = geometry.start + geometry.local_k;
  const int beta_warmup_end =
      (beta_start + warmup_length < k) ? (beta_start + warmup_length) : k;
  const int beta_warmup_k = beta_warmup_end - beta_start;

  extern __shared__ float scratch[];
  float* alpha_init = scratch;
  float* beta_init = alpha_init + kStates;
  float* alpha_window = beta_init + kStates;
  float* beta_window = alpha_window + (window_size + 1) * kStates;

  if (warmup_alpha_start == 0) {
    device_fill_state_row_tile8(alpha_init, kNegInf, tile);
    if (lane == 0) {
      alpha_init[0] = 0.0f;
    }
    tile.sync();
  } else {
    device_fill_state_row_tile8(alpha_init, 0.0f, tile);
  }

  if (warmup_alpha_k > 0) {
    device_stage2_window_forward_alpha_tile8(
        sys_llr, parity_llr, apriori_llr, warmup_alpha_start, warmup_alpha_k,
        alpha_init, alpha_window, tile);
    if (lane < kStates) {
      alpha_init[lane] = alpha_window[warmup_alpha_k * kStates + lane];
    }
    tile.sync();
  }

  if (beta_warmup_end == k) {
    if (lane == 0) {
      device_build_tail_beta(beta_init, tail_systematic_llr, tail_parity_llr);
    }
    tile.sync();
  } else {
    device_fill_state_row_tile8(beta_init, 0.0f, tile);
  }

  if (beta_warmup_k > 0) {
    device_stage2_backward_window_beta_tile8(
        sys_llr, parity_llr, apriori_llr, beta_start, beta_warmup_k,
        beta_init, beta_window, tile);
    if (lane < kStates) {
      beta_init[lane] = beta_window[lane];
    }
    tile.sync();
  }

  device_stage2_window_forward_alpha_tile8(
      sys_llr, parity_llr, apriori_llr, geometry.start, geometry.local_k,
      alpha_init, alpha_window, tile);
  device_stage2_backward_window_beta_tile8(
      sys_llr, parity_llr, apriori_llr, geometry.start, geometry.local_k,
      beta_init, beta_window, tile);

  for (int local_idx = lane; local_idx < geometry.local_k; local_idx += kStates) {
    const int global_idx = geometry.start + local_idx;
    sink[global_idx] = alpha_window[(local_idx + 1) * kStates + 0] +
                       beta_window[(local_idx + 1) * kStates + 0];
  }
}

__global__ void siso_parallel_window_approx_llr_experimental_kernel(
    const float* sys_llr,
    const float* parity_llr,
    const float* apriori_llr,
    const float* tail_systematic_llr,
    const float* tail_parity_llr,
    float* extrinsic,
    DeviceCodeBlockDescriptor descriptor,
    int window_size,
    int warmup_length) {
  const auto block = cg::this_thread_block();
  const auto tile = cg::tiled_partition<kStates>(block);
  if (tile.meta_group_rank() != 0) {
    return;
  }

  const int k = descriptor.k_r;
  const int window_count = stage2_window_count_for_k(k, window_size);
  const int window_index = static_cast<int>(blockIdx.x);
  const int lane = static_cast<int>(tile.thread_rank());
  if (window_index >= window_count) {
    return;
  }

  const Stage2WindowGeometry geometry =
      stage2_window_geometry(window_index, k, window_size);
  const int warmup_alpha_start =
      geometry.start > warmup_length ? (geometry.start - warmup_length) : 0;
  const int warmup_alpha_k = geometry.start - warmup_alpha_start;
  const int beta_start = geometry.start + geometry.local_k;
  const int beta_warmup_end =
      (beta_start + warmup_length < k) ? (beta_start + warmup_length) : k;
  const int beta_warmup_k = beta_warmup_end - beta_start;

  extern __shared__ float scratch[];
  float* alpha_init = scratch;
  float* beta_init = alpha_init + kStates;
  float* alpha_window = beta_init + kStates;
  float* beta_window = alpha_window + (window_size + 1) * kStates;

  if (warmup_alpha_start == 0) {
    device_fill_state_row_tile8(alpha_init, kNegInf, tile);
    if (lane == 0) {
      alpha_init[0] = 0.0f;
    }
    tile.sync();
  } else {
    device_fill_state_row_tile8(alpha_init, 0.0f, tile);
  }

  if (warmup_alpha_k > 0) {
    device_stage2_window_forward_alpha_tile8(
        sys_llr, parity_llr, apriori_llr, warmup_alpha_start, warmup_alpha_k,
        alpha_init, alpha_window, tile);
    if (lane < kStates) {
      alpha_init[lane] = alpha_window[warmup_alpha_k * kStates + lane];
    }
    tile.sync();
  }

  if (beta_warmup_end == k) {
    if (lane == 0) {
      device_build_tail_beta(beta_init, tail_systematic_llr, tail_parity_llr);
    }
    tile.sync();
  } else {
    device_fill_state_row_tile8(beta_init, 0.0f, tile);
  }

  if (beta_warmup_k > 0) {
    device_stage2_backward_window_beta_tile8(
        sys_llr, parity_llr, apriori_llr, beta_start, beta_warmup_k,
        beta_init, beta_window, tile);
    if (lane < kStates) {
      beta_init[lane] = beta_window[lane];
    }
    tile.sync();
  }

  device_stage2_window_forward_alpha_tile8(
      sys_llr, parity_llr, apriori_llr, geometry.start, geometry.local_k,
      alpha_init, alpha_window, tile);
  device_stage2_backward_window_beta_tile8(
      sys_llr, parity_llr, apriori_llr, geometry.start, geometry.local_k,
      beta_init, beta_window, tile);
  device_stage2_window_llr_tile8(
      sys_llr, parity_llr, apriori_llr, extrinsic, geometry.start, geometry.local_k,
      alpha_window, beta_window, tile);
}

std::vector<float> copy_to_host(const float* device_ptr, int count) {
  std::vector<float> host(count);
  TURBO_CUDA_CHECK(cudaMemcpy(host.data(), device_ptr, sizeof(float) * count, cudaMemcpyDeviceToHost));
  return host;
}

std::vector<float> turbo_decode_code_block_gpu_exact_common(const CodeBlockDescriptor& descriptor,
                                                            std::span<const float> d0_llr,
                                                            std::span<const float> d1_llr,
                                                            std::span<const float> d2_llr,
                                                            std::uint32_t max_iterations,
                                                            bool enable_early_stop,
                                                            bool* crc_ok,
                                                            std::uint32_t* iterations_used,
                                                            int /*window_size*/) {
  if (d0_llr.size() != descriptor.d_r || d1_llr.size() != descriptor.d_r || d2_llr.size() != descriptor.d_r) {
    throw std::invalid_argument("invalid LLR sizes for GPU Stage1 decode");
  }

  const int k = static_cast<int>(descriptor.k_r);
  const int d = static_cast<int>(descriptor.d_r);
  const auto perm = qpp_permutation(descriptor.k_r);
  const auto inv_perm = qpp_inverse_permutation(descriptor.k_r);

  std::vector<int> perm_int;
  std::vector<int> inv_perm_int;
  perm_int.reserve(perm.size());
  inv_perm_int.reserve(inv_perm.size());
  for (const auto value : perm) {
    perm_int.push_back(static_cast<int>(value));
  }
  for (const auto value : inv_perm) {
    inv_perm_int.push_back(static_cast<int>(value));
  }

  float *d_d0 = nullptr, *d_d1 = nullptr, *d_d2 = nullptr;
  float *d_sys_interleaved = nullptr, *d_apriori = nullptr, *d_ext1 = nullptr, *d_ext2 = nullptr;
  float *d_interleaved_apriori = nullptr, *d_ext2_interleaved = nullptr, *d_posterior = nullptr;
  float *d_upper_tail_sys = nullptr, *d_upper_tail_par = nullptr, *d_lower_tail_sys = nullptr, *d_lower_tail_par = nullptr;
  int *d_perm = nullptr, *d_inv_perm = nullptr;

  try {
    TURBO_CUDA_CHECK(cudaMalloc(&d_d0, sizeof(float) * d));
    TURBO_CUDA_CHECK(cudaMalloc(&d_d1, sizeof(float) * d));
    TURBO_CUDA_CHECK(cudaMalloc(&d_d2, sizeof(float) * d));
    TURBO_CUDA_CHECK(cudaMalloc(&d_sys_interleaved, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_apriori, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_ext1, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_ext2, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_interleaved_apriori, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_ext2_interleaved, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_posterior, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_upper_tail_sys, sizeof(float) * 4));
    TURBO_CUDA_CHECK(cudaMalloc(&d_upper_tail_par, sizeof(float) * 4));
    TURBO_CUDA_CHECK(cudaMalloc(&d_lower_tail_sys, sizeof(float) * 4));
    TURBO_CUDA_CHECK(cudaMalloc(&d_lower_tail_par, sizeof(float) * 4));
    TURBO_CUDA_CHECK(cudaMalloc(&d_perm, sizeof(int) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_inv_perm, sizeof(int) * k));

    TURBO_CUDA_CHECK(cudaMemcpy(d_d0, d0_llr.data(), sizeof(float) * d, cudaMemcpyHostToDevice));
    TURBO_CUDA_CHECK(cudaMemcpy(d_d1, d1_llr.data(), sizeof(float) * d, cudaMemcpyHostToDevice));
    TURBO_CUDA_CHECK(cudaMemcpy(d_d2, d2_llr.data(), sizeof(float) * d, cudaMemcpyHostToDevice));
    TURBO_CUDA_CHECK(cudaMemcpy(d_perm, perm_int.data(), sizeof(int) * k, cudaMemcpyHostToDevice));
    TURBO_CUDA_CHECK(cudaMemcpy(d_inv_perm, inv_perm_int.data(), sizeof(int) * k, cudaMemcpyHostToDevice));
    TURBO_CUDA_CHECK(cudaMemset(d_apriori, 0, sizeof(float) * k));

    const std::array<float, 4> upper_tail_systematic = {d0_llr[descriptor.k_r + 0], d0_llr[descriptor.k_r + 2], 0.0F, 0.0F};
    const std::array<float, 4> upper_tail_parity = {d1_llr[descriptor.k_r + 0], d0_llr[descriptor.k_r + 1], d1_llr[descriptor.k_r + 2], 0.0F};
    const std::array<float, 4> lower_tail_systematic = {d1_llr[descriptor.k_r + 3], d2_llr[descriptor.k_r + 0], d2_llr[descriptor.k_r + 2], 0.0F};
    const std::array<float, 4> lower_tail_parity = {d2_llr[descriptor.k_r + 1], d0_llr[descriptor.k_r + 3], d2_llr[descriptor.k_r + 3], 0.0F};

    TURBO_CUDA_CHECK(cudaMemcpy(d_upper_tail_sys, upper_tail_systematic.data(), sizeof(float) * 4, cudaMemcpyHostToDevice));
    TURBO_CUDA_CHECK(cudaMemcpy(d_upper_tail_par, upper_tail_parity.data(), sizeof(float) * 4, cudaMemcpyHostToDevice));
    TURBO_CUDA_CHECK(cudaMemcpy(d_lower_tail_sys, lower_tail_systematic.data(), sizeof(float) * 4, cudaMemcpyHostToDevice));
    TURBO_CUDA_CHECK(cudaMemcpy(d_lower_tail_par, lower_tail_parity.data(), sizeof(float) * 4, cudaMemcpyHostToDevice));

    const int threads = 128;
    const int blocks = (k + threads - 1) / threads;
    interleave_kernel<<<blocks, 128>>>(d_d0, d_perm, d_sys_interleaved, k);
    TURBO_CUDA_CHECK(cudaGetLastError());

    DeviceCodeBlockDescriptor device_descriptor{static_cast<int>(descriptor.k_r),
                                                static_cast<int>(descriptor.d_r),
                                                static_cast<int>(descriptor.filler_count)};

    const std::size_t shared_bytes = static_cast<std::size_t>(2 * (k + 1) * kStates) * sizeof(float);

    for (std::uint32_t iter = 0; iter < max_iterations; ++iter) {
      siso_exact_kernel<<<1, 1, shared_bytes>>>(d_d0, d_d1, d_apriori,
                                                d_upper_tail_sys, d_upper_tail_par,
                                                d_ext1, device_descriptor);
      TURBO_CUDA_CHECK(cudaGetLastError());

      interleave_kernel<<<blocks, threads>>>(d_ext1, d_perm, d_interleaved_apriori, k);
      TURBO_CUDA_CHECK(cudaGetLastError());

      siso_exact_kernel<<<1, 1, shared_bytes>>>(d_sys_interleaved, d_d2, d_interleaved_apriori,
                                                d_lower_tail_sys, d_lower_tail_par,
                                                d_ext2_interleaved, device_descriptor);
      TURBO_CUDA_CHECK(cudaGetLastError());

      deinterleave_kernel<<<blocks, threads>>>(d_ext2_interleaved, d_inv_perm, d_ext2, k);
      TURBO_CUDA_CHECK(cudaGetLastError());

      TURBO_CUDA_CHECK(cudaMemcpy(d_apriori, d_ext2, sizeof(float) * k, cudaMemcpyDeviceToDevice));
      posterior_kernel<<<blocks, threads>>>(d_d0, d_ext1, d_ext2, d_posterior, k);
      TURBO_CUDA_CHECK(cudaGetLastError());

      if (iterations_used != nullptr) {
        *iterations_used = iter + 1U;
      }
      (void)enable_early_stop;
    }

    std::vector<float> posterior = copy_to_host(d_posterior, k);
    if (crc_ok != nullptr) {
      *crc_ok = false;
    }

    cudaFree(d_d0);
    cudaFree(d_d1);
    cudaFree(d_d2);
    cudaFree(d_sys_interleaved);
    cudaFree(d_apriori);
    cudaFree(d_ext1);
    cudaFree(d_ext2);
    cudaFree(d_interleaved_apriori);
    cudaFree(d_ext2_interleaved);
    cudaFree(d_posterior);
    cudaFree(d_upper_tail_sys);
    cudaFree(d_upper_tail_par);
    cudaFree(d_lower_tail_sys);
    cudaFree(d_lower_tail_par);
    cudaFree(d_perm);
    cudaFree(d_inv_perm);
    return posterior;
  } catch (...) {
    cudaFree(d_d0);
    cudaFree(d_d1);
    cudaFree(d_d2);
    cudaFree(d_sys_interleaved);
    cudaFree(d_apriori);
    cudaFree(d_ext1);
    cudaFree(d_ext2);
    cudaFree(d_interleaved_apriori);
    cudaFree(d_ext2_interleaved);
    cudaFree(d_posterior);
    cudaFree(d_upper_tail_sys);
    cudaFree(d_upper_tail_par);
    cudaFree(d_lower_tail_sys);
    cudaFree(d_lower_tail_par);
    cudaFree(d_perm);
    cudaFree(d_inv_perm);
    throw;
  }
}

}  // namespace

std::vector<float> turbo_decode_code_block_gpu_stage2_parallel_approx_llr_experimental_cuda_impl(
    const CodeBlockDescriptor& descriptor,
    std::span<const float> d0_llr,
    std::span<const float> d1_llr,
    std::span<const float> d2_llr,
    std::uint32_t max_iterations,
    bool enable_early_stop,
    int warmup_length,
    bool* crc_ok,
    std::uint32_t* iterations_used);

std::vector<float> turbo_decode_code_block_gpu_stage1_cuda_impl(const CodeBlockDescriptor& descriptor,
                                                                std::span<const float> d0_llr,
                                                                std::span<const float> d1_llr,
                                                                std::span<const float> d2_llr,
                                                                std::uint32_t max_iterations,
                                                                bool enable_early_stop,
                                                                bool* crc_ok,
                                                                std::uint32_t* iterations_used) {
  return turbo_decode_code_block_gpu_exact_common(descriptor, d0_llr, d1_llr, d2_llr,
                                                  max_iterations, enable_early_stop,
                                                  crc_ok, iterations_used, descriptor.k_r);
}

std::vector<float> turbo_decode_code_block_gpu_stage1_tile8_experimental_cuda_impl(
    const CodeBlockDescriptor& descriptor,
    std::span<const float> d0_llr,
    std::span<const float> d1_llr,
    std::span<const float> d2_llr,
    std::uint32_t max_iterations,
    bool enable_early_stop,
    bool* crc_ok,
    std::uint32_t* iterations_used) {
  if (d0_llr.size() != descriptor.d_r || d1_llr.size() != descriptor.d_r || d2_llr.size() != descriptor.d_r) {
    throw std::invalid_argument("invalid LLR sizes for GPU experimental Stage1 decode");
  }

  const int k = static_cast<int>(descriptor.k_r);
  const int d = static_cast<int>(descriptor.d_r);
  const auto perm = qpp_permutation(descriptor.k_r);
  const auto inv_perm = qpp_inverse_permutation(descriptor.k_r);

  std::vector<int> perm_int;
  std::vector<int> inv_perm_int;
  perm_int.reserve(perm.size());
  inv_perm_int.reserve(inv_perm.size());
  for (const auto value : perm) {
    perm_int.push_back(static_cast<int>(value));
  }
  for (const auto value : inv_perm) {
    inv_perm_int.push_back(static_cast<int>(value));
  }

  float *d_d0 = nullptr, *d_d1 = nullptr, *d_d2 = nullptr;
  float *d_sys_interleaved = nullptr, *d_apriori = nullptr, *d_ext1 = nullptr, *d_ext2 = nullptr;
  float *d_interleaved_apriori = nullptr, *d_ext2_interleaved = nullptr, *d_posterior = nullptr;
  float *d_upper_tail_sys = nullptr, *d_upper_tail_par = nullptr, *d_lower_tail_sys = nullptr, *d_lower_tail_par = nullptr;
  int *d_perm = nullptr, *d_inv_perm = nullptr;
  cudaStream_t stream = nullptr;

  try {
    TURBO_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    TURBO_CUDA_CHECK(cudaMalloc(&d_d0, sizeof(float) * d));
    TURBO_CUDA_CHECK(cudaMalloc(&d_d1, sizeof(float) * d));
    TURBO_CUDA_CHECK(cudaMalloc(&d_d2, sizeof(float) * d));
    TURBO_CUDA_CHECK(cudaMalloc(&d_sys_interleaved, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_apriori, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_ext1, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_ext2, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_interleaved_apriori, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_ext2_interleaved, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_posterior, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_upper_tail_sys, sizeof(float) * 4));
    TURBO_CUDA_CHECK(cudaMalloc(&d_upper_tail_par, sizeof(float) * 4));
    TURBO_CUDA_CHECK(cudaMalloc(&d_lower_tail_sys, sizeof(float) * 4));
    TURBO_CUDA_CHECK(cudaMalloc(&d_lower_tail_par, sizeof(float) * 4));
    TURBO_CUDA_CHECK(cudaMalloc(&d_perm, sizeof(int) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_inv_perm, sizeof(int) * k));

    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_d0, d0_llr.data(), sizeof(float) * d, cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_d1, d1_llr.data(), sizeof(float) * d, cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_d2, d2_llr.data(), sizeof(float) * d, cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_perm, perm_int.data(), sizeof(int) * k, cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_inv_perm, inv_perm_int.data(), sizeof(int) * k, cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemsetAsync(d_apriori, 0, sizeof(float) * k, stream));

    const std::array<float, 4> upper_tail_systematic = {d0_llr[descriptor.k_r + 0], d0_llr[descriptor.k_r + 2], 0.0F, 0.0F};
    const std::array<float, 4> upper_tail_parity = {d1_llr[descriptor.k_r + 0], d0_llr[descriptor.k_r + 1], d1_llr[descriptor.k_r + 2], 0.0F};
    const std::array<float, 4> lower_tail_systematic = {d1_llr[descriptor.k_r + 3], d2_llr[descriptor.k_r + 0], d2_llr[descriptor.k_r + 2], 0.0F};
    const std::array<float, 4> lower_tail_parity = {d2_llr[descriptor.k_r + 1], d0_llr[descriptor.k_r + 3], d2_llr[descriptor.k_r + 3], 0.0F};

    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_upper_tail_sys, upper_tail_systematic.data(), sizeof(float) * 4, cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_upper_tail_par, upper_tail_parity.data(), sizeof(float) * 4, cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_lower_tail_sys, lower_tail_systematic.data(), sizeof(float) * 4, cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_lower_tail_par, lower_tail_parity.data(), sizeof(float) * 4, cudaMemcpyHostToDevice, stream));

    const int threads = 32;
    const int blocks = (k + 127) / 128;
    interleave_kernel<<<blocks, 128>>>(d_d0, d_perm, d_sys_interleaved, k);
    TURBO_CUDA_CHECK(cudaGetLastError());

    DeviceCodeBlockDescriptor device_descriptor{static_cast<int>(descriptor.k_r),
                                                static_cast<int>(descriptor.d_r),
                                                static_cast<int>(descriptor.filler_count)};
    const std::size_t shared_bytes = static_cast<std::size_t>(2 * (k + 1) * kStates) * sizeof(float);

    for (std::uint32_t iter = 0; iter < max_iterations; ++iter) {
      siso_exact_kernel_tile8_experimental<<<1, threads, shared_bytes>>>(
          d_d0, d_d1, d_apriori, d_upper_tail_sys, d_upper_tail_par, d_ext1, device_descriptor);
      TURBO_CUDA_CHECK(cudaGetLastError());

      interleave_kernel<<<blocks, 128>>>(d_ext1, d_perm, d_interleaved_apriori, k);
      TURBO_CUDA_CHECK(cudaGetLastError());

      siso_exact_kernel_tile8_experimental<<<1, threads, shared_bytes>>>(
          d_sys_interleaved, d_d2, d_interleaved_apriori, d_lower_tail_sys, d_lower_tail_par,
          d_ext2_interleaved, device_descriptor);
      TURBO_CUDA_CHECK(cudaGetLastError());

      deinterleave_kernel<<<blocks, 128>>>(d_ext2_interleaved, d_inv_perm, d_ext2, k);
      TURBO_CUDA_CHECK(cudaGetLastError());

      TURBO_CUDA_CHECK(cudaMemcpy(d_apriori, d_ext2, sizeof(float) * k, cudaMemcpyDeviceToDevice));
      posterior_kernel<<<blocks, 128>>>(d_d0, d_ext1, d_ext2, d_posterior, k);
      TURBO_CUDA_CHECK(cudaGetLastError());

      if (iterations_used != nullptr) {
        *iterations_used = iter + 1U;
      }
      (void)enable_early_stop;
    }

    std::vector<float> posterior = copy_to_host(d_posterior, k);
    if (crc_ok != nullptr) {
      *crc_ok = false;
    }

    cudaFree(d_d0);
    cudaFree(d_d1);
    cudaFree(d_d2);
    cudaFree(d_sys_interleaved);
    cudaFree(d_apriori);
    cudaFree(d_ext1);
    cudaFree(d_ext2);
    cudaFree(d_interleaved_apriori);
    cudaFree(d_ext2_interleaved);
    cudaFree(d_posterior);
    cudaFree(d_upper_tail_sys);
    cudaFree(d_upper_tail_par);
    cudaFree(d_lower_tail_sys);
    cudaFree(d_lower_tail_par);
    cudaFree(d_perm);
    cudaFree(d_inv_perm);
    return posterior;
  } catch (...) {
    cudaFree(d_d0);
    cudaFree(d_d1);
    cudaFree(d_d2);
    cudaFree(d_sys_interleaved);
    cudaFree(d_apriori);
    cudaFree(d_ext1);
    cudaFree(d_ext2);
    cudaFree(d_interleaved_apriori);
    cudaFree(d_ext2_interleaved);
    cudaFree(d_posterior);
    cudaFree(d_upper_tail_sys);
    cudaFree(d_upper_tail_par);
    cudaFree(d_lower_tail_sys);
    cudaFree(d_lower_tail_par);
    cudaFree(d_perm);
    cudaFree(d_inv_perm);
    throw;
  }
}

std::vector<float> turbo_decode_code_block_gpu_stage2_cuda_impl(const CodeBlockDescriptor& descriptor,
                                                                std::span<const float> d0_llr,
                                                                std::span<const float> d1_llr,
                                                                std::span<const float> d2_llr,
                                                                std::uint32_t max_iterations,
                                                                bool enable_early_stop,
                                                                bool* crc_ok,
                                                                std::uint32_t* iterations_used) {
  if (d0_llr.size() != descriptor.d_r || d1_llr.size() != descriptor.d_r || d2_llr.size() != descriptor.d_r) {
    throw std::invalid_argument("invalid LLR sizes for GPU Stage2 decode");
  }

  const int k = static_cast<int>(descriptor.k_r);
  const int d = static_cast<int>(descriptor.d_r);
  const auto perm = qpp_permutation(descriptor.k_r);
  const auto inv_perm = qpp_inverse_permutation(descriptor.k_r);

  std::vector<int> perm_int;
  std::vector<int> inv_perm_int;
  perm_int.reserve(perm.size());
  inv_perm_int.reserve(inv_perm.size());
  for (const auto value : perm) {
    perm_int.push_back(static_cast<int>(value));
  }
  for (const auto value : inv_perm) {
    inv_perm_int.push_back(static_cast<int>(value));
  }

  float *d_d0 = nullptr, *d_d1 = nullptr, *d_d2 = nullptr;
  float *d_sys_interleaved = nullptr, *d_apriori = nullptr, *d_ext1 = nullptr, *d_ext2 = nullptr;
  float *d_interleaved_apriori = nullptr, *d_ext2_interleaved = nullptr, *d_posterior = nullptr;
  float *d_upper_tail_sys = nullptr, *d_upper_tail_par = nullptr, *d_lower_tail_sys = nullptr, *d_lower_tail_par = nullptr;
  int *d_perm = nullptr, *d_inv_perm = nullptr;
  cudaStream_t stream = nullptr;

  try {
    TURBO_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    TURBO_CUDA_CHECK(cudaMalloc(&d_d0, sizeof(float) * d));
    TURBO_CUDA_CHECK(cudaMalloc(&d_d1, sizeof(float) * d));
    TURBO_CUDA_CHECK(cudaMalloc(&d_d2, sizeof(float) * d));
    TURBO_CUDA_CHECK(cudaMalloc(&d_sys_interleaved, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_apriori, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_ext1, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_ext2, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_interleaved_apriori, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_ext2_interleaved, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_posterior, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_upper_tail_sys, sizeof(float) * 4));
    TURBO_CUDA_CHECK(cudaMalloc(&d_upper_tail_par, sizeof(float) * 4));
    TURBO_CUDA_CHECK(cudaMalloc(&d_lower_tail_sys, sizeof(float) * 4));
    TURBO_CUDA_CHECK(cudaMalloc(&d_lower_tail_par, sizeof(float) * 4));
    TURBO_CUDA_CHECK(cudaMalloc(&d_perm, sizeof(int) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_inv_perm, sizeof(int) * k));

    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_d0, d0_llr.data(), sizeof(float) * d, cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_d1, d1_llr.data(), sizeof(float) * d, cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_d2, d2_llr.data(), sizeof(float) * d, cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_perm, perm_int.data(), sizeof(int) * k, cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_inv_perm, inv_perm_int.data(), sizeof(int) * k, cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemsetAsync(d_apriori, 0, sizeof(float) * k, stream));

    const std::array<float, 4> upper_tail_systematic = {d0_llr[descriptor.k_r + 0], d0_llr[descriptor.k_r + 2], 0.0F, 0.0F};
    const std::array<float, 4> upper_tail_parity = {d1_llr[descriptor.k_r + 0], d0_llr[descriptor.k_r + 1], d1_llr[descriptor.k_r + 2], 0.0F};
    const std::array<float, 4> lower_tail_systematic = {d1_llr[descriptor.k_r + 3], d2_llr[descriptor.k_r + 0], d2_llr[descriptor.k_r + 2], 0.0F};
    const std::array<float, 4> lower_tail_parity = {d2_llr[descriptor.k_r + 1], d0_llr[descriptor.k_r + 3], d2_llr[descriptor.k_r + 3], 0.0F};

    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_upper_tail_sys, upper_tail_systematic.data(), sizeof(float) * 4, cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_upper_tail_par, upper_tail_parity.data(), sizeof(float) * 4, cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_lower_tail_sys, lower_tail_systematic.data(), sizeof(float) * 4, cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_lower_tail_par, lower_tail_parity.data(), sizeof(float) * 4, cudaMemcpyHostToDevice, stream));

    const int threads = 128;
    const int blocks = (k + threads - 1) / threads;
    interleave_kernel<<<blocks, threads, 0, stream>>>(d_d0, d_perm, d_sys_interleaved, k);
    TURBO_CUDA_CHECK(cudaGetLastError());

    DeviceCodeBlockDescriptor device_descriptor{static_cast<int>(descriptor.k_r),
                                                static_cast<int>(descriptor.d_r),
                                                static_cast<int>(descriptor.filler_count)};
    const Stage2LaunchConfig launch =
        make_stage2_launch_config(k, kStage2WindowSize);

    // P2b dual-path: prefer the stored-alpha kernel (one fewer forward pass) when its
    // shared-memory footprint fits the device limit; otherwise fall back to the
    // windowed kernel, which keeps only bounded per-window scratch and scales to large K.
    const std::size_t stored_alpha_shared =
        stage2_stored_alpha_shared_bytes(k, kStage2WindowSize);
    int shared_limit_per_block = 0;
    {
      int device_id = 0;
      TURBO_CUDA_CHECK(cudaGetDevice(&device_id));
      TURBO_CUDA_CHECK(cudaDeviceGetAttribute(
          &shared_limit_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device_id));
    }
    const bool use_stored_alpha =
        stored_alpha_shared <= static_cast<std::size_t>(shared_limit_per_block);
    if (!use_stored_alpha) {
      return turbo_decode_code_block_gpu_stage2_parallel_approx_llr_experimental_cuda_impl(
          descriptor, d0_llr, d1_llr, d2_llr, max_iterations, enable_early_stop,
          16, crc_ok, iterations_used);
    }
    const std::size_t stage2_shared =
        stored_alpha_shared;

    auto launch_stage2 = [&](const float* sys, const float* par, const float* ap,
                             const float* tail_sys, const float* tail_par, float* ext) {
      siso_exact_stored_alpha_kernel<<<1, kStates, stage2_shared, stream>>>(
          sys, par, ap, tail_sys, tail_par, ext, device_descriptor, launch.window_size);
    };

    for (std::uint32_t iter = 0; iter < max_iterations; ++iter) {
      launch_stage2(d_d0, d_d1, d_apriori, d_upper_tail_sys, d_upper_tail_par, d_ext1);
      TURBO_CUDA_CHECK(cudaGetLastError());

      interleave_kernel<<<blocks, threads, 0, stream>>>(d_ext1, d_perm, d_interleaved_apriori, k);
      TURBO_CUDA_CHECK(cudaGetLastError());

      launch_stage2(d_sys_interleaved, d_d2, d_interleaved_apriori,
                    d_lower_tail_sys, d_lower_tail_par, d_ext2_interleaved);
      TURBO_CUDA_CHECK(cudaGetLastError());

      deinterleave_kernel<<<blocks, threads, 0, stream>>>(d_ext2_interleaved, d_inv_perm, d_ext2, k);
      TURBO_CUDA_CHECK(cudaGetLastError());

      TURBO_CUDA_CHECK(cudaMemcpyAsync(d_apriori, d_ext2, sizeof(float) * k, cudaMemcpyDeviceToDevice, stream));
      posterior_kernel<<<blocks, threads, 0, stream>>>(d_d0, d_ext1, d_ext2, d_posterior, k);
      TURBO_CUDA_CHECK(cudaGetLastError());

      if (iterations_used != nullptr) {
        *iterations_used = iter + 1U;
      }
      (void)enable_early_stop;
    }

    TURBO_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> posterior = copy_to_host(d_posterior, k);
    if (crc_ok != nullptr) {
      *crc_ok = false;
    }

    cudaFree(d_d0);
    cudaFree(d_d1);
    cudaFree(d_d2);
    cudaFree(d_sys_interleaved);
    cudaFree(d_apriori);
    cudaFree(d_ext1);
    cudaFree(d_ext2);
    cudaFree(d_interleaved_apriori);
    cudaFree(d_ext2_interleaved);
    cudaFree(d_posterior);
    cudaFree(d_upper_tail_sys);
    cudaFree(d_upper_tail_par);
    cudaFree(d_lower_tail_sys);
    cudaFree(d_lower_tail_par);
    cudaFree(d_perm);
    cudaFree(d_inv_perm);
    cudaStreamDestroy(stream);
    return posterior;
  } catch (...) {
    cudaFree(d_d0);
    cudaFree(d_d1);
    cudaFree(d_d2);
    cudaFree(d_sys_interleaved);
    cudaFree(d_apriori);
    cudaFree(d_ext1);
    cudaFree(d_ext2);
    cudaFree(d_interleaved_apriori);
    cudaFree(d_ext2_interleaved);
    cudaFree(d_posterior);
    cudaFree(d_upper_tail_sys);
    cudaFree(d_upper_tail_par);
    cudaFree(d_lower_tail_sys);
    cudaFree(d_lower_tail_par);
    cudaFree(d_perm);
    cudaFree(d_inv_perm);
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    throw;
  }
}

std::vector<float> turbo_decode_code_block_gpu_stage2_parallel_shell_experimental_cuda_impl(
    const CodeBlockDescriptor& descriptor,
    std::span<const float> d0_llr,
    std::span<const float> d1_llr,
    std::span<const float> d2_llr,
    std::uint32_t max_iterations,
    bool enable_early_stop,
    bool* crc_ok,
    std::uint32_t* iterations_used) {
  if (d0_llr.size() != descriptor.d_r || d1_llr.size() != descriptor.d_r ||
      d2_llr.size() != descriptor.d_r) {
    throw std::invalid_argument(
        "invalid LLR sizes for GPU Stage2 parallel-shell experimental decode");
  }

  const int k = static_cast<int>(descriptor.k_r);
  const int d = static_cast<int>(descriptor.d_r);
  float *d_d0 = nullptr, *d_d1 = nullptr, *d_apriori = nullptr, *d_ext = nullptr;
  cudaStream_t stream = nullptr;

  try {
    TURBO_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    TURBO_CUDA_CHECK(cudaMalloc(&d_d0, sizeof(float) * d));
    TURBO_CUDA_CHECK(cudaMalloc(&d_d1, sizeof(float) * d));
    TURBO_CUDA_CHECK(cudaMalloc(&d_apriori, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_ext, sizeof(float) * k));

    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_d0, d0_llr.data(), sizeof(float) * d,
                                     cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_d1, d1_llr.data(), sizeof(float) * d,
                                     cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemsetAsync(d_apriori, 0, sizeof(float) * k, stream));
    TURBO_CUDA_CHECK(cudaMemsetAsync(d_ext, 0, sizeof(float) * k, stream));

    DeviceCodeBlockDescriptor device_descriptor{static_cast<int>(descriptor.k_r),
                                                static_cast<int>(descriptor.d_r),
                                                static_cast<int>(descriptor.filler_count)};
    const Stage2LaunchConfig launch =
        make_stage2_launch_config(k, kStage2WindowSize);

    for (std::uint32_t iter = 0; iter < max_iterations; ++iter) {
      siso_parallel_window_shell_kernel<<<launch.window_count,
                                          kStage2ParallelShellBlockSize,
                                          0,
                                          stream>>>(
          d_d0, d_d1, d_apriori, d_ext, device_descriptor, launch.window_size);
      TURBO_CUDA_CHECK(cudaGetLastError());
      if (iterations_used != nullptr) {
        *iterations_used = iter + 1U;
      }
    }

    TURBO_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> posterior = copy_to_host(d_ext, k);
    if (crc_ok != nullptr) {
      *crc_ok = false;
    }
    (void)d2_llr;
    (void)enable_early_stop;

    cudaFree(d_d0);
    cudaFree(d_d1);
    cudaFree(d_apriori);
    cudaFree(d_ext);
    cudaStreamDestroy(stream);
    return posterior;
  } catch (...) {
    cudaFree(d_d0);
    cudaFree(d_d1);
    cudaFree(d_apriori);
    cudaFree(d_ext);
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    throw;
  }
}

std::vector<float> turbo_decode_code_block_gpu_stage2_parallel_forward_experimental_cuda_impl(
    const CodeBlockDescriptor& descriptor,
    std::span<const float> d0_llr,
    std::span<const float> d1_llr,
    std::span<const float> d2_llr,
    std::uint32_t max_iterations,
    bool enable_early_stop,
    bool* crc_ok,
    std::uint32_t* iterations_used) {
  if (d0_llr.size() != descriptor.d_r || d1_llr.size() != descriptor.d_r ||
      d2_llr.size() != descriptor.d_r) {
    throw std::invalid_argument(
        "invalid LLR sizes for GPU Stage2 parallel-forward experimental decode");
  }

  const int k = static_cast<int>(descriptor.k_r);
  const int d = static_cast<int>(descriptor.d_r);
  float *d_d0 = nullptr, *d_d1 = nullptr, *d_apriori = nullptr, *d_sink = nullptr;
  cudaStream_t stream = nullptr;

  try {
    TURBO_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    TURBO_CUDA_CHECK(cudaMalloc(&d_d0, sizeof(float) * d));
    TURBO_CUDA_CHECK(cudaMalloc(&d_d1, sizeof(float) * d));
    TURBO_CUDA_CHECK(cudaMalloc(&d_apriori, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float) * k));

    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_d0, d0_llr.data(), sizeof(float) * d,
                                     cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_d1, d1_llr.data(), sizeof(float) * d,
                                     cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemsetAsync(d_apriori, 0, sizeof(float) * k, stream));
    TURBO_CUDA_CHECK(cudaMemsetAsync(d_sink, 0, sizeof(float) * k, stream));

    DeviceCodeBlockDescriptor device_descriptor{static_cast<int>(descriptor.k_r),
                                                static_cast<int>(descriptor.d_r),
                                                static_cast<int>(descriptor.filler_count)};
    const Stage2LaunchConfig launch =
        make_stage2_launch_config(k, kStage2WindowSize);
    const std::size_t shared_bytes =
        static_cast<std::size_t>((kStage2WindowSize + 2) * kStates) * sizeof(float);

    for (std::uint32_t iter = 0; iter < max_iterations; ++iter) {
      siso_parallel_window_forward_experimental_kernel<<<launch.window_count,
                                                         kStage2ParallelShellBlockSize,
                                                         shared_bytes,
                                                         stream>>>(
          d_d0, d_d1, d_apriori, d_sink, device_descriptor,
          launch.window_size, kStage2ParallelForwardWarmup);
      TURBO_CUDA_CHECK(cudaGetLastError());
      if (iterations_used != nullptr) {
        *iterations_used = iter + 1U;
      }
    }

    TURBO_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> posterior = copy_to_host(d_sink, k);
    if (crc_ok != nullptr) {
      *crc_ok = false;
    }
    (void)d2_llr;
    (void)enable_early_stop;

    cudaFree(d_d0);
    cudaFree(d_d1);
    cudaFree(d_apriori);
    cudaFree(d_sink);
    cudaStreamDestroy(stream);
    return posterior;
  } catch (...) {
    cudaFree(d_d0);
    cudaFree(d_d1);
    cudaFree(d_apriori);
    cudaFree(d_sink);
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    throw;
  }
}

std::vector<float> turbo_decode_code_block_gpu_stage2_parallel_forward_backward_experimental_cuda_impl(
    const CodeBlockDescriptor& descriptor,
    std::span<const float> d0_llr,
    std::span<const float> d1_llr,
    std::span<const float> d2_llr,
    std::uint32_t max_iterations,
    bool enable_early_stop,
    bool* crc_ok,
    std::uint32_t* iterations_used) {
  if (d0_llr.size() != descriptor.d_r || d1_llr.size() != descriptor.d_r ||
      d2_llr.size() != descriptor.d_r) {
    throw std::invalid_argument(
        "invalid LLR sizes for GPU Stage2 parallel-forward-backward experimental decode");
  }

  const int k = static_cast<int>(descriptor.k_r);
  const int d = static_cast<int>(descriptor.d_r);
  float *d_d0 = nullptr, *d_d1 = nullptr, *d_apriori = nullptr, *d_sink = nullptr;
  float *d_upper_tail_sys = nullptr, *d_upper_tail_par = nullptr;
  cudaStream_t stream = nullptr;

  try {
    TURBO_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    TURBO_CUDA_CHECK(cudaMalloc(&d_d0, sizeof(float) * d));
    TURBO_CUDA_CHECK(cudaMalloc(&d_d1, sizeof(float) * d));
    TURBO_CUDA_CHECK(cudaMalloc(&d_apriori, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_upper_tail_sys, sizeof(float) * 4));
    TURBO_CUDA_CHECK(cudaMalloc(&d_upper_tail_par, sizeof(float) * 4));

    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_d0, d0_llr.data(), sizeof(float) * d,
                                     cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_d1, d1_llr.data(), sizeof(float) * d,
                                     cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemsetAsync(d_apriori, 0, sizeof(float) * k, stream));
    TURBO_CUDA_CHECK(cudaMemsetAsync(d_sink, 0, sizeof(float) * k, stream));

    const std::array<float, 4> upper_tail_systematic = {d0_llr[descriptor.k_r + 0],
                                                        d0_llr[descriptor.k_r + 2], 0.0F, 0.0F};
    const std::array<float, 4> upper_tail_parity = {d1_llr[descriptor.k_r + 0],
                                                    d0_llr[descriptor.k_r + 1],
                                                    d1_llr[descriptor.k_r + 2], 0.0F};
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_upper_tail_sys, upper_tail_systematic.data(),
                                     sizeof(float) * 4, cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_upper_tail_par, upper_tail_parity.data(),
                                     sizeof(float) * 4, cudaMemcpyHostToDevice, stream));

    DeviceCodeBlockDescriptor device_descriptor{static_cast<int>(descriptor.k_r),
                                                static_cast<int>(descriptor.d_r),
                                                static_cast<int>(descriptor.filler_count)};
    const Stage2LaunchConfig launch =
        make_stage2_launch_config(k, kStage2WindowSize);
    const std::size_t shared_bytes =
        static_cast<std::size_t>((2 * (kStage2WindowSize + 1) + 2) * kStates) * sizeof(float);

    for (std::uint32_t iter = 0; iter < max_iterations; ++iter) {
      siso_parallel_window_forward_backward_experimental_kernel<<<launch.window_count,
                                                                  kStage2ParallelShellBlockSize,
                                                                  shared_bytes,
                                                                  stream>>>(
          d_d0, d_d1, d_apriori, d_upper_tail_sys, d_upper_tail_par, d_sink,
          device_descriptor, launch.window_size, kStage2ParallelForwardWarmup);
      TURBO_CUDA_CHECK(cudaGetLastError());
      if (iterations_used != nullptr) {
        *iterations_used = iter + 1U;
      }
    }

    TURBO_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> posterior = copy_to_host(d_sink, k);
    if (crc_ok != nullptr) {
      *crc_ok = false;
    }
    (void)d2_llr;
    (void)enable_early_stop;

    cudaFree(d_d0);
    cudaFree(d_d1);
    cudaFree(d_apriori);
    cudaFree(d_sink);
    cudaFree(d_upper_tail_sys);
    cudaFree(d_upper_tail_par);
    cudaStreamDestroy(stream);
    return posterior;
  } catch (...) {
    cudaFree(d_d0);
    cudaFree(d_d1);
    cudaFree(d_apriori);
    cudaFree(d_sink);
    cudaFree(d_upper_tail_sys);
    cudaFree(d_upper_tail_par);
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    throw;
  }
}

std::vector<float> turbo_decode_code_block_gpu_stage2_parallel_approx_llr_experimental_cuda_impl(
    const CodeBlockDescriptor& descriptor,
    std::span<const float> d0_llr,
    std::span<const float> d1_llr,
    std::span<const float> d2_llr,
    std::uint32_t max_iterations,
    bool enable_early_stop,
    int warmup_length,
    bool* crc_ok,
    std::uint32_t* iterations_used) {
  if (d0_llr.size() != descriptor.d_r || d1_llr.size() != descriptor.d_r ||
      d2_llr.size() != descriptor.d_r) {
    throw std::invalid_argument(
        "invalid LLR sizes for GPU Stage2 parallel-approx-llr experimental decode");
  }

  const int k = static_cast<int>(descriptor.k_r);
  const int d = static_cast<int>(descriptor.d_r);
  float *d_d0 = nullptr, *d_d1 = nullptr, *d_apriori = nullptr, *d_ext = nullptr;
  float *d_upper_tail_sys = nullptr, *d_upper_tail_par = nullptr;
  cudaStream_t stream = nullptr;

  try {
    TURBO_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    TURBO_CUDA_CHECK(cudaMalloc(&d_d0, sizeof(float) * d));
    TURBO_CUDA_CHECK(cudaMalloc(&d_d1, sizeof(float) * d));
    TURBO_CUDA_CHECK(cudaMalloc(&d_apriori, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_ext, sizeof(float) * k));
    TURBO_CUDA_CHECK(cudaMalloc(&d_upper_tail_sys, sizeof(float) * 4));
    TURBO_CUDA_CHECK(cudaMalloc(&d_upper_tail_par, sizeof(float) * 4));

    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_d0, d0_llr.data(), sizeof(float) * d,
                                     cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_d1, d1_llr.data(), sizeof(float) * d,
                                     cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemsetAsync(d_apriori, 0, sizeof(float) * k, stream));
    TURBO_CUDA_CHECK(cudaMemsetAsync(d_ext, 0, sizeof(float) * k, stream));

    const std::array<float, 4> upper_tail_systematic = {d0_llr[descriptor.k_r + 0],
                                                        d0_llr[descriptor.k_r + 2], 0.0F, 0.0F};
    const std::array<float, 4> upper_tail_parity = {d1_llr[descriptor.k_r + 0],
                                                    d0_llr[descriptor.k_r + 1],
                                                    d1_llr[descriptor.k_r + 2], 0.0F};
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_upper_tail_sys, upper_tail_systematic.data(),
                                     sizeof(float) * 4, cudaMemcpyHostToDevice, stream));
    TURBO_CUDA_CHECK(cudaMemcpyAsync(d_upper_tail_par, upper_tail_parity.data(),
                                     sizeof(float) * 4, cudaMemcpyHostToDevice, stream));

    DeviceCodeBlockDescriptor device_descriptor{static_cast<int>(descriptor.k_r),
                                                static_cast<int>(descriptor.d_r),
                                                static_cast<int>(descriptor.filler_count)};
    const Stage2LaunchConfig launch =
        make_stage2_launch_config(k, kStage2WindowSize);
    const std::size_t shared_bytes =
        static_cast<std::size_t>((2 * (kStage2WindowSize + 1) + 2) * kStates) * sizeof(float);

    for (std::uint32_t iter = 0; iter < max_iterations; ++iter) {
      siso_parallel_window_approx_llr_experimental_kernel<<<launch.window_count,
                                                            kStage2ParallelShellBlockSize,
                                                            shared_bytes,
                                                            stream>>>(
          d_d0, d_d1, d_apriori, d_upper_tail_sys, d_upper_tail_par, d_ext,
          device_descriptor, launch.window_size, warmup_length);
      TURBO_CUDA_CHECK(cudaGetLastError());
      if (iterations_used != nullptr) {
        *iterations_used = iter + 1U;
      }
    }

    TURBO_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> posterior = copy_to_host(d_ext, k);
    if (crc_ok != nullptr) {
      *crc_ok = false;
    }
    (void)d2_llr;
    (void)enable_early_stop;

    cudaFree(d_d0);
    cudaFree(d_d1);
    cudaFree(d_apriori);
    cudaFree(d_ext);
    cudaFree(d_upper_tail_sys);
    cudaFree(d_upper_tail_par);
    cudaStreamDestroy(stream);
    return posterior;
  } catch (...) {
    cudaFree(d_d0);
    cudaFree(d_d1);
    cudaFree(d_apriori);
    cudaFree(d_ext);
    cudaFree(d_upper_tail_sys);
    cudaFree(d_upper_tail_par);
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
    throw;
  }
}

}  // namespace turbo_cpp

#endif
