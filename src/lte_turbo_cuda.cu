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

__global__ void siso_exact_windowed_kernel(const float* sys_llr,
                                           const float* parity_llr,
                                           const float* apriori_llr,
                                           const float* tail_systematic_llr,
                                           const float* tail_parity_llr,
                                           float* extrinsic,
                                           DeviceCodeBlockDescriptor descriptor,
                                           int window_size) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int k = descriptor.k_r;
  const int window_count = (k + window_size - 1) / window_size;

  extern __shared__ float scratch[];
  float* alpha_in = scratch;
  float* beta_in = alpha_in + window_count * kStates;
  float* alpha_window = beta_in + window_count * kStates;
  float* beta_window = alpha_window + (window_size + 1) * kStates;

  for (int i = 0; i < window_count * kStates; ++i) {
    alpha_in[i] = kNegInf;
    beta_in[i] = kNegInf;
  }
  alpha_in[0] = 0.0f;
  device_build_tail_beta(&beta_in[(window_count - 1) * kStates], tail_systematic_llr, tail_parity_llr);

  for (int w = 0; w < window_count; ++w) {
    const int start = w * window_size;
    const int end = min(start + window_size, k);
    const int local_k = end - start;

    for (int s = 0; s < kStates; ++s) {
      alpha_window[s] = alpha_in[w * kStates + s];
    }
    for (int idx = 0; idx < local_k; ++idx) {
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
          const int global_idx = start + idx;
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
    if (w + 1 < window_count) {
      for (int s = 0; s < kStates; ++s) {
        alpha_in[(w + 1) * kStates + s] = alpha_window[local_k * kStates + s];
      }
    }
  }

  for (int w = window_count - 1; w >= 0; --w) {
    const int start = w * window_size;
    const int end = min(start + window_size, k);
    const int local_k = end - start;

    for (int s = 0; s < kStates; ++s) {
      beta_window[local_k * kStates + s] = beta_in[w * kStates + s];
    }
    for (int idx = local_k - 1; idx >= 0; --idx) {
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
          const int global_idx = start + idx;
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
    if (w > 0) {
      for (int s = 0; s < kStates; ++s) {
        beta_in[(w - 1) * kStates + s] = beta_window[s];
      }
    }

    for (int s = 0; s < kStates; ++s) {
      alpha_window[s] = alpha_in[w * kStates + s];
    }
    for (int idx = 0; idx < local_k; ++idx) {
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
          const int global_idx = start + idx;
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

    for (int idx = 0; idx < local_k; ++idx) {
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
          const int global_idx = start + idx;
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
      extrinsic[start + idx] = (llr1 - llr0) - sys_llr[start + idx] - apriori_llr[start + idx];
    }
  }
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
    interleave_kernel<<<blocks, threads>>>(d_d0, d_perm, d_sys_interleaved, k);
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
    interleave_kernel<<<blocks, threads>>>(d_d0, d_perm, d_sys_interleaved, k);
    TURBO_CUDA_CHECK(cudaGetLastError());

    DeviceCodeBlockDescriptor device_descriptor{static_cast<int>(descriptor.k_r),
                                                static_cast<int>(descriptor.d_r),
                                                static_cast<int>(descriptor.filler_count)};
    const int window_size = 128;
    const int window_count = (k + window_size - 1) / window_size;
    const std::size_t shared_bytes =
        static_cast<std::size_t>((2 * window_count + 2 * (window_size + 1)) * kStates) * sizeof(float);

    for (std::uint32_t iter = 0; iter < max_iterations; ++iter) {
      siso_exact_windowed_kernel<<<1, 1, shared_bytes>>>(d_d0, d_d1, d_apriori,
                                                         d_upper_tail_sys, d_upper_tail_par,
                                                         d_ext1, device_descriptor, window_size);
      TURBO_CUDA_CHECK(cudaGetLastError());

      interleave_kernel<<<blocks, threads>>>(d_ext1, d_perm, d_interleaved_apriori, k);
      TURBO_CUDA_CHECK(cudaGetLastError());

      siso_exact_windowed_kernel<<<1, 1, shared_bytes>>>(d_sys_interleaved, d_d2, d_interleaved_apriori,
                                                         d_lower_tail_sys, d_lower_tail_par,
                                                         d_ext2_interleaved, device_descriptor, window_size);
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

}  // namespace turbo_cpp

#endif
