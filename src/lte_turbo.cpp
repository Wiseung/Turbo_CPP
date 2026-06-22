#include "turbo_cpp/lte_turbo.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>

namespace turbo_cpp {

#if TURBO_CPP_HAS_CUDA
std::vector<float> turbo_decode_code_block_gpu_stage1_cuda_impl(const CodeBlockDescriptor& descriptor,
                                                                std::span<const float> d0_llr,
                                                                std::span<const float> d1_llr,
                                                                std::span<const float> d2_llr,
                                                                std::uint32_t max_iterations,
                                                                bool enable_early_stop,
                                                                bool* crc_ok,
                                                                std::uint32_t* iterations_used);
std::vector<float> turbo_decode_code_block_gpu_stage2_cuda_impl(const CodeBlockDescriptor& descriptor,
                                                                std::span<const float> d0_llr,
                                                                std::span<const float> d1_llr,
                                                                std::span<const float> d2_llr,
                                                                std::uint32_t max_iterations,
                                                                bool enable_early_stop,
                                                                bool* crc_ok,
                                                                std::uint32_t* iterations_used);
std::vector<float> turbo_decode_code_block_gpu_stage1_tile8_experimental_cuda_impl(
    const CodeBlockDescriptor& descriptor,
    std::span<const float> d0_llr,
    std::span<const float> d1_llr,
    std::span<const float> d2_llr,
    std::uint32_t max_iterations,
    bool enable_early_stop,
    bool* crc_ok,
    std::uint32_t* iterations_used);
#endif

namespace {

constexpr std::size_t kMaxCodeBlockSize = 6144;
constexpr std::size_t kMinTurboBlockSize = 40;
constexpr std::size_t kSubblockColumns = 32;
constexpr std::array<std::size_t, 32> kInterColumnPattern = {
    0, 16, 8,  24, 4,  20, 12, 28, 2,  18, 10, 26, 6,  22, 14, 30,
    1, 17, 9,  25, 5,  21, 13, 29, 3,  19, 11, 27, 7,  23, 15, 31,
};

constexpr std::array<QppEntry, 188> kQppTable = {{
    {40, 3, 10},   {48, 7, 12},   {56, 19, 42},  {64, 7, 16},
    {72, 7, 18},   {80, 11, 20},  {88, 5, 22},   {96, 11, 24},
    {104, 7, 26},  {112, 41, 84}, {120, 103, 90}, {128, 15, 32},
    {136, 9, 34},  {144, 17, 108}, {152, 9, 38}, {160, 21, 120},
    {168, 101, 84}, {176, 21, 44}, {184, 57, 46}, {192, 23, 48},
    {200, 13, 50}, {208, 27, 52}, {216, 11, 36}, {224, 27, 56},
    {232, 85, 58}, {240, 29, 60}, {248, 33, 62}, {256, 15, 32},
    {264, 17, 198}, {272, 33, 68}, {280, 103, 210}, {288, 19, 36},
    {296, 19, 74}, {304, 37, 76}, {312, 19, 78}, {320, 21, 120},
    {328, 21, 82}, {336, 115, 84}, {344, 193, 86}, {352, 21, 44},
    {360, 133, 90}, {368, 81, 46}, {376, 45, 94}, {384, 23, 48},
    {392, 243, 98}, {400, 151, 40}, {408, 155, 102}, {416, 25, 52},
    {424, 51, 106}, {432, 47, 72}, {440, 91, 110}, {448, 29, 168},
    {456, 29, 114}, {464, 247, 58}, {472, 29, 118}, {480, 89, 180},
    {488, 91, 122}, {496, 157, 62}, {504, 55, 84}, {512, 31, 64},
    {528, 17, 66}, {544, 35, 68}, {560, 227, 420}, {576, 65, 96},
    {592, 19, 74}, {608, 37, 76}, {624, 41, 234}, {640, 39, 80},
    {656, 185, 82}, {672, 43, 252}, {688, 21, 86}, {704, 155, 44},
    {720, 79, 120}, {736, 139, 92}, {752, 23, 94}, {768, 217, 48},
    {784, 25, 98}, {800, 17, 80}, {816, 127, 102}, {832, 25, 52},
    {848, 239, 106}, {864, 17, 48}, {880, 137, 110}, {896, 215, 112},
    {912, 29, 114}, {928, 15, 58}, {944, 147, 118}, {960, 29, 60},
    {976, 59, 122}, {992, 65, 124}, {1008, 55, 84}, {1024, 31, 64},
    {1056, 17, 66}, {1088, 171, 204}, {1120, 67, 140}, {1152, 35, 72},
    {1184, 19, 74}, {1216, 39, 76}, {1248, 19, 78}, {1280, 199, 240},
    {1312, 21, 82}, {1344, 211, 252}, {1376, 21, 86}, {1408, 43, 88},
    {1440, 149, 60}, {1472, 45, 92}, {1504, 49, 846}, {1536, 71, 48},
    {1568, 13, 28}, {1600, 17, 80}, {1632, 25, 102}, {1664, 183, 104},
    {1696, 55, 954}, {1728, 127, 96}, {1760, 27, 110}, {1792, 29, 112},
    {1824, 29, 114}, {1856, 57, 116}, {1888, 45, 354}, {1920, 31, 120},
    {1952, 59, 610}, {1984, 185, 124}, {2016, 113, 420}, {2048, 31, 64},
    {2112, 17, 66}, {2176, 171, 136}, {2240, 209, 420}, {2304, 253, 216},
    {2368, 367, 444}, {2432, 265, 456}, {2496, 181, 468}, {2560, 39, 80},
    {2624, 27, 164}, {2688, 127, 504}, {2752, 143, 172}, {2816, 43, 88},
    {2880, 29, 300}, {2944, 45, 92}, {3008, 157, 188}, {3072, 47, 96},
    {3136, 13, 28}, {3200, 111, 240}, {3264, 443, 204}, {3328, 51, 104},
    {3392, 51, 212}, {3456, 451, 192}, {3520, 257, 220}, {3584, 57, 336},
    {3648, 313, 228}, {3712, 271, 232}, {3776, 179, 236}, {3840, 331, 120},
    {3904, 363, 244}, {3968, 375, 248}, {4032, 127, 168}, {4096, 31, 64},
    {4160, 33, 130}, {4224, 43, 264}, {4288, 33, 134}, {4352, 477, 408},
    {4416, 35, 138}, {4480, 233, 280}, {4544, 357, 142}, {4608, 337, 480},
    {4672, 37, 146}, {4736, 71, 444}, {4800, 71, 120}, {4864, 37, 152},
    {4928, 39, 462}, {4992, 127, 234}, {5056, 39, 158}, {5120, 39, 80},
    {5184, 31, 96}, {5248, 113, 902}, {5312, 41, 166}, {5376, 251, 336},
    {5440, 43, 170}, {5504, 21, 86}, {5568, 43, 174}, {5632, 45, 176},
    {5696, 45, 178}, {5760, 161, 120}, {5824, 89, 182}, {5888, 323, 184},
    {5952, 47, 186}, {6016, 23, 94}, {6080, 47, 190}, {6144, 263, 480},
}};

constexpr std::array<std::uint32_t, 24> kCrc24APoly = {
    1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
    1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1,
};
constexpr std::array<std::uint32_t, 24> kCrc24BPoly = {
    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1,
};
constexpr std::array<std::uint32_t, 16> kCrc16Poly = {
    1, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 1,
};
constexpr std::array<std::uint32_t, 8> kCrc8Poly = {1, 1, 0, 1, 1, 0, 0, 1};

template <std::size_t N>
std::vector<std::uint8_t> append_crc(std::span<const std::uint8_t> bits,
                                     const std::array<std::uint32_t, N>& poly) {
  std::vector<std::uint8_t> work(bits.begin(), bits.end());
  work.resize(bits.size() + N, 0);
  for (std::size_t i = 0; i < bits.size(); ++i) {
    if ((work[i] & 1U) == 0U) {
      continue;
    }
    work[i] = 0;
    for (std::size_t j = 0; j < N; ++j) {
      work[i + 1 + j] ^= static_cast<std::uint8_t>(poly[j]);
    }
  }
  std::vector<std::uint8_t> out(bits.begin(), bits.end());
  out.insert(out.end(), work.end() - static_cast<std::ptrdiff_t>(N), work.end());
  return out;
}

template <std::size_t N>
bool check_crc(std::span<const std::uint8_t> bits_with_crc,
               const std::array<std::uint32_t, N>& poly) {
  if (bits_with_crc.size() < N) {
    return false;
  }
  std::vector<std::uint8_t> work(bits_with_crc.begin(), bits_with_crc.end());
  for (std::size_t i = 0; i + N < work.size(); ++i) {
    if ((work[i] & 1U) == 0U) {
      continue;
    }
    work[i] = 0;
    for (std::size_t j = 0; j < N; ++j) {
      work[i + 1 + j] ^= static_cast<std::uint8_t>(poly[j]);
    }
  }
  return std::all_of(work.end() - static_cast<std::ptrdiff_t>(N), work.end(),
                     [](std::uint8_t bit) { return bit == 0; });
}

const QppEntry* find_qpp(std::size_t k) {
  const auto it = std::find_if(kQppTable.begin(), kQppTable.end(),
                               [k](const QppEntry& entry) { return entry.k == k; });
  return it == kQppTable.end() ? nullptr : &(*it);
}

std::size_t ceil_div(std::size_t a, std::size_t b) {
  return (a + b - 1U) / b;
}

std::size_t compute_e_for_block(std::size_t r, std::size_t g_total,
                                std::size_t c, std::size_t num_layers,
                                std::size_t mod_order) {
  if (c == 0) {
    throw std::invalid_argument("code block count must be positive");
  }
  if (num_layers == 0 || mod_order == 0) {
    throw std::invalid_argument("num_layers and mod_order must be positive");
  }
  const std::size_t nl_qm = num_layers * mod_order;
  if (g_total % nl_qm != 0) {
    throw std::invalid_argument("g_total must be divisible by num_layers * mod_order");
  }
  const std::size_t g_prime = g_total / nl_qm;
  const std::size_t gamma = g_prime % c;
  const std::size_t low = nl_qm * (g_prime / c);
  const std::size_t high = nl_qm * ceil_div(g_prime, c);
  return (r <= c - gamma - 1U) ? low : high;
}

struct RateMatchLayout {
  std::size_t d = 0;
  std::size_t rows = 0;
  std::size_t k_pi = 0;
  std::size_t n_cb = 0;
  std::array<std::vector<int>, 3> maps;
};

std::vector<int> make_d01_subblock_map(std::size_t d, std::size_t filler_count) {
  const std::size_t rows = ceil_div(d, kSubblockColumns);
  const std::size_t k_pi = rows * kSubblockColumns;
  const std::size_t dummy = k_pi - d;
  std::vector<int> matrix(k_pi, -1);
  for (std::size_t i = 0; i < d; ++i) {
    const int src_index = (i < filler_count) ? -1 : static_cast<int>(i);
    matrix[dummy + i] = src_index;
  }
  std::vector<int> permuted(k_pi, -1);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < kSubblockColumns; ++col) {
      permuted[row * kSubblockColumns + col] =
          matrix[row * kSubblockColumns + kInterColumnPattern[col]];
    }
  }
  std::vector<int> out(k_pi, -1);
  std::size_t pos = 0;
  for (std::size_t col = 0; col < kSubblockColumns; ++col) {
    for (std::size_t row = 0; row < rows; ++row) {
      out[pos++] = permuted[row * kSubblockColumns + col];
    }
  }
  return out;
}

std::vector<int> make_d2_subblock_map(std::size_t d) {
  const std::size_t rows = ceil_div(d, kSubblockColumns);
  const std::size_t k_pi = rows * kSubblockColumns;
  const std::size_t dummy = k_pi - d;
  std::vector<int> y(k_pi, -1);
  for (std::size_t i = 0; i < d; ++i) {
    y[dummy + i] = static_cast<int>(i);
  }
  std::vector<int> out(k_pi, -1);
  for (std::size_t k = 0; k < k_pi; ++k) {
    const std::size_t index =
        kInterColumnPattern[k / rows] + kSubblockColumns * (k % rows) + 1U;
    out[k] = y[index % k_pi];
  }
  return out;
}

RateMatchLayout build_rate_match_layout(std::size_t d, std::size_t filler_count) {
  RateMatchLayout layout;
  layout.d = d;
  layout.rows = ceil_div(d, kSubblockColumns);
  layout.k_pi = layout.rows * kSubblockColumns;
  layout.n_cb = 3 * layout.k_pi;
  layout.maps[0] = make_d01_subblock_map(d, filler_count);
  layout.maps[1] = make_d01_subblock_map(d, filler_count);
  layout.maps[2] = make_d2_subblock_map(d);
  return layout;
}

std::vector<std::uint8_t> puncture_and_collect(const RateMatchLayout& layout,
                                               std::span<const std::uint8_t> systematic,
                                               std::span<const std::uint8_t> parity0,
                                               std::span<const std::uint8_t> parity1,
                                               std::size_t e_r, std::size_t rv) {
  std::vector<int> circular(layout.n_cb, -1);
  for (std::size_t i = 0; i < layout.k_pi; ++i) {
    circular[i] = layout.maps[0][i] < 0 ? -1 : static_cast<int>(systematic[layout.maps[0][i]]);
    circular[layout.k_pi + 2 * i] =
        layout.maps[1][i] < 0 ? -1 : static_cast<int>(parity0[layout.maps[1][i]]);
    circular[layout.k_pi + 2 * i + 1] =
        layout.maps[2][i] < 0 ? -1 : static_cast<int>(parity1[layout.maps[2][i]]);
  }
  const std::size_t k0 =
      layout.rows * (2 * ceil_div(layout.n_cb, 8 * layout.rows) * rv + 2);
  std::vector<std::uint8_t> out;
  out.reserve(e_r);
  std::size_t k = 0;
  std::size_t j = 0;
  while (k < e_r) {
    const int candidate = circular[(k0 + j) % layout.n_cb];
    if (candidate >= 0) {
      out.push_back(static_cast<std::uint8_t>(candidate));
      ++k;
    }
    ++j;
  }
  return out;
}

std::array<std::vector<float>, 3> derate_match_streams(const RateMatchLayout& layout,
                                                       std::span<const std::uint8_t> rate_matched_bits,
                                                       std::size_t rv) {
  std::vector<float> circular(layout.n_cb, 0.0F);
  std::vector<bool> valid(layout.n_cb, false);
  auto circular_is_valid_index = [&layout](std::size_t idx) {
    if (idx < layout.k_pi) {
      return layout.maps[0][idx] >= 0;
    }
    const std::size_t local = idx - layout.k_pi;
    const std::size_t lane = local / 2;
    const bool even = (local % 2) == 0;
    return even ? (layout.maps[1][lane] >= 0) : (layout.maps[2][lane] >= 0);
  };
  const std::size_t k0 =
      layout.rows * (2 * ceil_div(layout.n_cb, 8 * layout.rows) * rv + 2);
  std::size_t k = 0;
  std::size_t j = 0;
  while (k < rate_matched_bits.size()) {
    const std::size_t idx = (k0 + j) % layout.n_cb;
    if (circular_is_valid_index(idx)) {
      circular[idx] += rate_matched_bits[k] ? 8.0F : -8.0F;
      valid[idx] = true;
      ++k;
    }
    ++j;
  }

  std::array<std::vector<float>, 3> out = {
      std::vector<float>(layout.d, 0.0F),
      std::vector<float>(layout.d, 0.0F),
      std::vector<float>(layout.d, 0.0F),
  };
  for (std::size_t i = 0; i < layout.k_pi; ++i) {
    if (layout.maps[0][i] >= 0 && valid[i]) {
      out[0][layout.maps[0][i]] += circular[i];
    }
    if (layout.maps[1][i] >= 0 && valid[layout.k_pi + 2 * i]) {
      out[1][layout.maps[1][i]] += circular[layout.k_pi + 2 * i];
    }
    if (layout.maps[2][i] >= 0 && valid[layout.k_pi + 2 * i + 1]) {
      out[2][layout.maps[2][i]] += circular[layout.k_pi + 2 * i + 1];
    }
  }
  return out;
}

struct RscStep {
  std::uint8_t systematic = 0;
  std::uint8_t parity = 0;
  std::uint8_t next_state = 0;
};

RscStep rsc_encode_bit(std::uint8_t state, std::uint8_t input) {
  const std::uint8_t s0 = state & 1U;
  const std::uint8_t s1 = (state >> 1U) & 1U;
  const std::uint8_t s2 = (state >> 2U) & 1U;
  const std::uint8_t feedback = input ^ s1 ^ s2;
  const std::uint8_t parity = feedback ^ s0 ^ s2;
  const std::uint8_t next_state = static_cast<std::uint8_t>((feedback << 2U) | (s2 << 1U) | s1);
  return {.systematic = input, .parity = parity, .next_state = next_state};
}

std::array<std::uint8_t, 3> termination_inputs(std::uint8_t state) {
  std::array<std::uint8_t, 3> inputs{};
  for (std::size_t i = 0; i < 3; ++i) {
    const std::uint8_t s1 = (state >> 1U) & 1U;
    const std::uint8_t s2 = (state >> 2U) & 1U;
    const std::uint8_t input = static_cast<std::uint8_t>(s1 ^ s2);
    inputs[i] = input;
    state = rsc_encode_bit(state, input).next_state;
  }
  return inputs;
}

void normalize_state_metrics(std::array<float, 8>& metrics);

std::array<float, 8> build_tail_beta_initial(std::span<const float, 4> tail_systematic_llr,
                                             std::span<const float, 4> tail_parity_llr) {
  std::array<float, 8> beta{};
  beta.fill(-std::numeric_limits<float>::infinity());
  for (std::size_t state = 0; state < 8; ++state) {
    float metric = 0.0F;
    std::uint8_t current = static_cast<std::uint8_t>(state);
    const auto inputs = termination_inputs(current);
    for (std::size_t step = 0; step < 3; ++step) {
      const auto rsc = rsc_encode_bit(current, inputs[step]);
      current = rsc.next_state;
      const float sys_term = 0.5F * ((inputs[step] ? 1.0F : -1.0F) * tail_systematic_llr[step]);
      const float par_term = 0.5F * ((rsc.parity ? 1.0F : -1.0F) * tail_parity_llr[step]);
      metric += sys_term + par_term;
    }
    if (current == 0U) {
      beta[state] = metric;
    }
  }
  normalize_state_metrics(beta);
  return beta;
}

float max_star(float x, float y) {
  if (!std::isfinite(x)) {
    return y;
  }
  if (!std::isfinite(y)) {
    return x;
  }
  const float hi = std::max(x, y);
  const float lo = std::min(x, y);
  return hi + std::log1pf(std::exp(lo - hi));
}

void normalize_state_metrics(std::array<float, 8>& metrics) {
  const float mx = *std::max_element(metrics.begin(), metrics.end());
  if (!std::isfinite(mx)) {
    return;
  }
  for (float& value : metrics) {
    if (std::isfinite(value)) {
      value -= mx;
    }
  }
}

std::vector<float> decode_siso_log_map(std::span<const float> sys_llr,
                                       std::span<const float> parity_llr,
                                       std::span<const float> apriori_llr,
                                       std::span<const float, 4> tail_systematic_llr,
                                       std::span<const float, 4> tail_parity_llr) {
  const std::size_t k = sys_llr.size();
  if (parity_llr.size() != k + 4 || apriori_llr.size() != k) {
    throw std::invalid_argument("invalid SISO input dimensions");
  }
  std::vector<std::array<float, 8>> alpha(k + 1);
  std::vector<std::array<float, 8>> beta(k + 1);
  for (auto& row : alpha) {
    row.fill(-std::numeric_limits<float>::infinity());
  }
  for (auto& row : beta) {
    row.fill(-std::numeric_limits<float>::infinity());
  }
  alpha[0][0] = 0.0F;
  beta[k] = build_tail_beta_initial(tail_systematic_llr, tail_parity_llr);

  for (std::size_t idx = 0; idx < k; ++idx) {
    for (std::size_t prev = 0; prev < 8; ++prev) {
      if (!std::isfinite(alpha[idx][prev])) {
        continue;
      }
      for (std::uint8_t u = 0; u < 2; ++u) {
        const auto step = rsc_encode_bit(static_cast<std::uint8_t>(prev), u);
        const float gamma = 0.5F * ((u ? 1.0F : -1.0F) * (sys_llr[idx] + apriori_llr[idx]) +
                                    (step.parity ? 1.0F : -1.0F) * parity_llr[idx]);
        alpha[idx + 1][step.next_state] =
            std::isfinite(alpha[idx + 1][step.next_state])
                ? max_star(alpha[idx + 1][step.next_state], alpha[idx][prev] + gamma)
                : alpha[idx][prev] + gamma;
      }
    }
    normalize_state_metrics(alpha[idx + 1]);
  }

  for (std::size_t step_idx = k; step_idx-- > 0;) {
    for (std::size_t prev = 0; prev < 8; ++prev) {
      for (std::uint8_t u = 0; u < 2; ++u) {
        const auto step = rsc_encode_bit(static_cast<std::uint8_t>(prev), u);
        if (!std::isfinite(beta[step_idx + 1][step.next_state])) {
          continue;
        }
        const float gamma = 0.5F * ((u ? 1.0F : -1.0F) * (sys_llr[step_idx] + apriori_llr[step_idx]) +
                                    (step.parity ? 1.0F : -1.0F) * parity_llr[step_idx]);
        beta[step_idx][prev] =
            std::isfinite(beta[step_idx][prev])
                ? max_star(beta[step_idx][prev], beta[step_idx + 1][step.next_state] + gamma)
                : beta[step_idx + 1][step.next_state] + gamma;
      }
    }
    normalize_state_metrics(beta[step_idx]);
  }

  std::vector<float> extrinsic(k, 0.0F);
  for (std::size_t idx = 0; idx < k; ++idx) {
    float llr0 = -std::numeric_limits<float>::infinity();
    float llr1 = -std::numeric_limits<float>::infinity();
    for (std::size_t prev = 0; prev < 8; ++prev) {
      for (std::uint8_t u = 0; u < 2; ++u) {
        const auto step = rsc_encode_bit(static_cast<std::uint8_t>(prev), u);
        const float gamma = 0.5F * ((u ? 1.0F : -1.0F) * (sys_llr[idx] + apriori_llr[idx]) +
                                    (step.parity ? 1.0F : -1.0F) * parity_llr[idx]);
        const float metric = alpha[idx][prev] + gamma + beta[idx + 1][step.next_state];
        if (u == 0U) {
          llr0 = std::isfinite(llr0) ? max_star(llr0, metric) : metric;
        } else {
          llr1 = std::isfinite(llr1) ? max_star(llr1, metric) : metric;
        }
      }
    }
    extrinsic[idx] = (llr1 - llr0) - sys_llr[idx] - apriori_llr[idx];
  }
  return extrinsic;
}

std::vector<std::uint8_t> hard_decision(std::span<const float> posterior_llr) {
  std::vector<std::uint8_t> bits(posterior_llr.size(), 0);
  for (std::size_t i = 0; i < posterior_llr.size(); ++i) {
    bits[i] = posterior_llr[i] >= 0.0F ? 1U : 0U;
  }
  return bits;
}

}  // namespace

std::span<const QppEntry> qpp_interleaver_table() { return kQppTable; }

const QppEntry& qpp_entry_for_k(std::size_t k) {
  const QppEntry* entry = find_qpp(k);
  if (entry == nullptr) {
    throw std::out_of_range("unsupported QPP block size");
  }
  return *entry;
}

std::vector<std::uint8_t> append_crc24a(std::span<const std::uint8_t> bits) {
  return append_crc(bits, kCrc24APoly);
}

std::vector<std::uint8_t> append_crc24b(std::span<const std::uint8_t> bits) {
  return append_crc(bits, kCrc24BPoly);
}

std::vector<std::uint8_t> append_crc16(std::span<const std::uint8_t> bits) {
  return append_crc(bits, kCrc16Poly);
}

std::vector<std::uint8_t> append_crc8(std::span<const std::uint8_t> bits) {
  return append_crc(bits, kCrc8Poly);
}

bool check_crc24a(std::span<const std::uint8_t> bits_with_crc) {
  return check_crc(bits_with_crc, kCrc24APoly);
}

bool check_crc24b(std::span<const std::uint8_t> bits_with_crc) {
  return check_crc(bits_with_crc, kCrc24BPoly);
}

bool check_crc16(std::span<const std::uint8_t> bits_with_crc) {
  return check_crc(bits_with_crc, kCrc16Poly);
}

bool check_crc8(std::span<const std::uint8_t> bits_with_crc) {
  return check_crc(bits_with_crc, kCrc8Poly);
}

SegmentationResult segment_transport_block(std::span<const std::uint8_t> bits_with_crc) {
  if (bits_with_crc.empty()) {
    throw std::invalid_argument("transport block cannot be empty");
  }
  const std::size_t b = bits_with_crc.size();
  SegmentationResult result;
  std::size_t b_prime = b;
  if (b <= kMaxCodeBlockSize) {
    result.l = 0;
    result.c = 1;
  } else {
    result.l = 24;
    result.c = ceil_div(b, kMaxCodeBlockSize - result.l);
    b_prime = b + result.c * result.l;
  }

  const auto upper_it = std::find_if(kQppTable.begin(), kQppTable.end(),
                                     [b_prime, &result](const QppEntry& entry) {
                                       return result.c * entry.k >= b_prime;
                                     });
  if (upper_it == kQppTable.end()) {
    throw std::runtime_error("failed to select K+");
  }
  result.k_plus = upper_it->k;
  if (result.c == 1) {
    result.c_plus = 1;
    result.k_minus = 0;
    result.c_minus = 0;
  } else {
    const auto lower_it = std::find_if(kQppTable.rbegin(), kQppTable.rend(),
                                       [k_plus = result.k_plus](const QppEntry& entry) {
                                         return entry.k < k_plus;
                                       });
    if (lower_it == kQppTable.rend()) {
      throw std::runtime_error("failed to select K-");
    }
    result.k_minus = lower_it->k;
    const std::size_t delta_k = result.k_plus - result.k_minus;
    result.c_minus = (result.c * result.k_plus - b_prime) / delta_k;
    result.c_plus = result.c - result.c_minus;
  }
  result.filler_bits = result.c_plus * result.k_plus + result.c_minus * result.k_minus - b_prime;
  result.code_blocks.reserve(result.c);

  std::size_t cursor = 0;
  for (std::size_t r = 0; r < result.c; ++r) {
    const std::size_t k_r = (r < result.c_minus) ? result.k_minus : result.k_plus;
    std::vector<std::uint8_t> block(k_r, 0);
    std::size_t offset = 0;
    if (r == 0 && result.filler_bits > 0) {
      offset = result.filler_bits;
    }
    const std::size_t payload_bits = k_r - offset - result.l;
    std::copy_n(bits_with_crc.begin() + static_cast<std::ptrdiff_t>(cursor),
                static_cast<std::ptrdiff_t>(payload_bits), block.begin() + static_cast<std::ptrdiff_t>(offset));
    cursor += payload_bits;
    if (result.l == 24) {
      const auto with_crc = append_crc24b(std::span<const std::uint8_t>(block).subspan(offset, payload_bits));
      std::copy(with_crc.begin() + static_cast<std::ptrdiff_t>(payload_bits), with_crc.end(),
                block.begin() + static_cast<std::ptrdiff_t>(offset + payload_bits));
    }
    result.code_blocks.push_back(std::move(block));
  }
  return result;
}

std::vector<std::size_t> qpp_permutation(std::size_t k) {
  const auto& entry = qpp_entry_for_k(k);
  std::vector<std::size_t> perm(k);
  for (std::size_t i = 0; i < k; ++i) {
    perm[i] = (entry.f1 * i + entry.f2 * i * i) % k;
  }
  return perm;
}

std::vector<std::size_t> qpp_inverse_permutation(std::size_t k) {
  const auto perm = qpp_permutation(k);
  std::vector<std::size_t> inverse(k);
  for (std::size_t i = 0; i < k; ++i) {
    inverse[perm[i]] = i;
  }
  return inverse;
}

EncodedCodeBlock turbo_encode_code_block(const CodeBlockDescriptor& descriptor,
                                         std::span<const std::uint8_t> cb_bits) {
  if (cb_bits.size() != descriptor.k_r) {
    throw std::invalid_argument("code block length mismatch");
  }
  const auto permutation = qpp_permutation(descriptor.k_r);
  EncodedCodeBlock out;
  out.descriptor = descriptor;
  out.cb_bits.assign(cb_bits.begin(), cb_bits.end());
  out.systematic.reserve(descriptor.k_r + 4);
  out.parity0.reserve(descriptor.k_r + 4);
  out.parity1.reserve(descriptor.k_r + 4);

  std::uint8_t upper_state = 0;
  std::uint8_t lower_state = 0;

  for (std::size_t i = 0; i < descriptor.k_r; ++i) {
    const std::uint8_t input0 = cb_bits[i];
    const auto upper = rsc_encode_bit(upper_state, input0);
    upper_state = upper.next_state;
    out.systematic.push_back(i < descriptor.filler_count ? 0U : upper.systematic);
    out.parity0.push_back(i < descriptor.filler_count ? 0U : upper.parity);

    const std::uint8_t input1 = cb_bits[permutation[i]];
    const auto lower = rsc_encode_bit(lower_state, input1);
    lower_state = lower.next_state;
    out.parity1.push_back(lower.parity);
  }

  const auto upper_tail_inputs = termination_inputs(upper_state);
  const auto lower_tail_inputs = termination_inputs(lower_state);
  std::array<RscStep, 3> upper_tail{};
  std::array<RscStep, 3> lower_tail{};
  for (std::size_t i = 0; i < 3; ++i) {
    upper_tail[i] = rsc_encode_bit(upper_state, upper_tail_inputs[i]);
    upper_state = upper_tail[i].next_state;
    lower_tail[i] = rsc_encode_bit(lower_state, lower_tail_inputs[i]);
    lower_state = lower_tail[i].next_state;
  }

  out.systematic.push_back(upper_tail[0].systematic);
  out.systematic.push_back(upper_tail[1].parity);
  out.systematic.push_back(upper_tail[1].systematic);
  out.systematic.push_back(upper_tail[2].parity);

  out.parity0.push_back(upper_tail[0].parity);
  out.parity0.push_back(upper_tail[1].systematic);
  out.parity0.push_back(upper_tail[2].parity);
  out.parity0.push_back(lower_tail[0].systematic);

  out.parity1.push_back(lower_tail[0].parity);
  out.parity1.push_back(lower_tail[1].systematic);
  out.parity1.push_back(lower_tail[1].parity);
  out.parity1.push_back(lower_tail[2].systematic);

  for (std::size_t i = 0; i < 3; ++i) {
    out.upper_tail_systematic[i] = upper_tail[i].systematic;
    out.upper_tail_parity[i] = upper_tail[i].parity;
    out.lower_tail_systematic[i] = lower_tail[i].systematic;
    out.lower_tail_parity[i] = lower_tail[i].parity;
  }

  out.descriptor.d_r = out.systematic.size();
  return out;
}

EncodedTransportBlock encode_transport_block(const LteTurboCodecParams& params,
                                             std::span<const std::uint8_t> transport_bits) {
  if (params.transport_block_bits != transport_bits.size()) {
    throw std::invalid_argument("transport block size does not match params");
  }
  EncodedTransportBlock out;
  out.params = params;
  out.transport_bits.assign(transport_bits.begin(), transport_bits.end());
  out.tb_bits_with_crc = append_crc24a(transport_bits);

  const SegmentationResult segmentation = segment_transport_block(out.tb_bits_with_crc);
  out.code_blocks.reserve(segmentation.code_blocks.size());
  for (std::size_t r = 0; r < segmentation.code_blocks.size(); ++r) {
    const auto& bits = segmentation.code_blocks[r];
    const auto& qpp = qpp_entry_for_k(bits.size());
    CodeBlockDescriptor descriptor;
    descriptor.block_index = r;
    descriptor.k_r = bits.size();
    descriptor.d_r = bits.size() + 4;
    descriptor.e_r = compute_e_for_block(r, params.g_total, segmentation.c, params.num_layers, params.mod_order);
    descriptor.filler_count = (r == 0) ? segmentation.filler_bits : 0;
    descriptor.qpp_index = static_cast<std::size_t>(&qpp - kQppTable.data());
    auto encoded = turbo_encode_code_block(descriptor, bits);
    const auto layout = build_rate_match_layout(descriptor.d_r, descriptor.filler_count);
    encoded.rate_matched_bits =
        puncture_and_collect(layout, encoded.systematic, encoded.parity0, encoded.parity1,
                             descriptor.e_r, params.rv);
    out.code_blocks.push_back(std::move(encoded));
  }
  return out;
}

std::vector<float> derate_match_code_block(const LteTurboCodecParams& params,
                                           const CodeBlockDescriptor& descriptor,
                                           std::span<const std::uint8_t> rate_matched_bits) {
  const auto layout = build_rate_match_layout(descriptor.d_r, descriptor.filler_count);
  const auto streams = derate_match_streams(layout, rate_matched_bits, params.rv);

  std::vector<float> llrs;
  llrs.reserve(3 * descriptor.d_r);
  llrs.insert(llrs.end(), streams[0].begin(), streams[0].end());  // d(0)
  llrs.insert(llrs.end(), streams[1].begin(), streams[1].end());  // d(1)
  llrs.insert(llrs.end(), streams[2].begin(), streams[2].end());  // d(2)
  return llrs;
}

std::vector<float> turbo_decode_code_block_cpu(const CodeBlockDescriptor& descriptor,
                                               std::span<const float> d0_llr,
                                               std::span<const float> d1_llr,
                                               std::span<const float> d2_llr,
                                               std::uint32_t max_iterations,
                                               bool enable_early_stop,
                                               bool* crc_ok,
                                               std::uint32_t* iterations_used) {
  if (d0_llr.size() != descriptor.d_r || d1_llr.size() != descriptor.d_r ||
      d2_llr.size() != descriptor.d_r) {
    throw std::invalid_argument("invalid LLR sizes for turbo decode");
  }
  const std::span<const float> sys_llr(d0_llr.data(), descriptor.k_r);
  std::vector<float> apriori(descriptor.k_r, 0.0F);
  std::vector<float> extrinsic1(descriptor.k_r, 0.0F);
  std::vector<float> extrinsic2(descriptor.k_r, 0.0F);
  const auto perm = qpp_permutation(descriptor.k_r);
  const auto inv_perm = qpp_inverse_permutation(descriptor.k_r);
  std::vector<float> sys_interleaved(descriptor.k_r, 0.0F);
  for (std::size_t i = 0; i < descriptor.k_r; ++i) {
    sys_interleaved[i] = sys_llr[perm[i]];
  }
  const std::array<float, 4> upper_tail_systematic = {d0_llr[descriptor.k_r + 0],
                                                       d0_llr[descriptor.k_r + 2],
                                                       0.0F,
                                                       0.0F};
  const std::array<float, 4> upper_tail_parity = {d1_llr[descriptor.k_r + 0],
                                                   d0_llr[descriptor.k_r + 1],
                                                   d1_llr[descriptor.k_r + 2],
                                                   0.0F};
  const std::array<float, 4> lower_tail_systematic = {d1_llr[descriptor.k_r + 3],
                                                       d2_llr[descriptor.k_r + 0],
                                                       d2_llr[descriptor.k_r + 2],
                                                       0.0F};
  const std::array<float, 4> lower_tail_parity = {d2_llr[descriptor.k_r + 1],
                                                   d0_llr[descriptor.k_r + 3],
                                                   d2_llr[descriptor.k_r + 3],
                                                   0.0F};

  bool cb_crc_ok = false;
  std::vector<float> posterior(descriptor.k_r, 0.0F);
  for (std::uint32_t iter = 0; iter < max_iterations; ++iter) {
    extrinsic1 = decode_siso_log_map(sys_llr, d1_llr, apriori,
                                     upper_tail_systematic, upper_tail_parity);
    std::vector<float> interleaved_apriori(descriptor.k_r, 0.0F);
    for (std::size_t i = 0; i < descriptor.k_r; ++i) {
      interleaved_apriori[i] = extrinsic1[perm[i]];
    }
    const auto ext2_interleaved = decode_siso_log_map(sys_interleaved, d2_llr, interleaved_apriori,
                                                      lower_tail_systematic, lower_tail_parity);
    for (std::size_t i = 0; i < descriptor.k_r; ++i) {
      extrinsic2[i] = ext2_interleaved[inv_perm[i]];
      apriori[i] = extrinsic2[i];
      posterior[i] = sys_llr[i] + extrinsic1[i] + extrinsic2[i];
    }

    if (enable_early_stop) {
      auto decoded = hard_decision(posterior);
      if (descriptor.filler_count > 0) {
        std::fill(decoded.begin(), decoded.begin() + static_cast<std::ptrdiff_t>(descriptor.filler_count), 0U);
      }
      cb_crc_ok = check_crc24b(std::span<const std::uint8_t>(decoded).subspan(descriptor.filler_count));
      if (cb_crc_ok) {
        if (iterations_used != nullptr) {
          *iterations_used = iter + 1U;
        }
        break;
      }
    }
    if (iterations_used != nullptr) {
      *iterations_used = iter + 1U;
    }
  }
  if (crc_ok != nullptr) {
    *crc_ok = cb_crc_ok;
  }
  return posterior;
}

std::vector<float> turbo_decode_code_block_gpu_stage1(const CodeBlockDescriptor& descriptor,
                                                      std::span<const float> d0_llr,
                                                      std::span<const float> d1_llr,
                                                      std::span<const float> d2_llr,
                                                      std::uint32_t max_iterations,
                                                      bool enable_early_stop,
                                                      bool* crc_ok,
                                                      std::uint32_t* iterations_used) {
#if TURBO_CPP_HAS_CUDA
  return turbo_decode_code_block_gpu_stage1_tile8_experimental_cuda_impl(
      descriptor, d0_llr, d1_llr, d2_llr, max_iterations, enable_early_stop, crc_ok,
      iterations_used);
#else
  (void)descriptor;
  (void)d0_llr;
  (void)d1_llr;
  (void)d2_llr;
  (void)max_iterations;
  (void)enable_early_stop;
  (void)crc_ok;
  (void)iterations_used;
  throw std::runtime_error("GPU Stage 1 requested but CUDA support is not built");
#endif
}

std::vector<float> turbo_decode_code_block_gpu_stage2(const CodeBlockDescriptor& descriptor,
                                                      std::span<const float> d0_llr,
                                                      std::span<const float> d1_llr,
                                                      std::span<const float> d2_llr,
                                                      std::uint32_t max_iterations,
                                                      bool enable_early_stop,
                                                      bool* crc_ok,
                                                      std::uint32_t* iterations_used) {
#if TURBO_CPP_HAS_CUDA
  return turbo_decode_code_block_gpu_stage2_cuda_impl(descriptor, d0_llr, d1_llr, d2_llr,
                                                      max_iterations, enable_early_stop, crc_ok,
                                                      iterations_used);
#else
  (void)descriptor;
  (void)d0_llr;
  (void)d1_llr;
  (void)d2_llr;
  (void)max_iterations;
  (void)enable_early_stop;
  (void)crc_ok;
  (void)iterations_used;
  throw std::runtime_error("GPU Stage 2 requested but CUDA support is not built");
#endif
}

std::vector<float> turbo_decode_code_block_gpu_stage1_tile8_experimental(
    const CodeBlockDescriptor& descriptor,
    std::span<const float> d0_llr,
    std::span<const float> d1_llr,
    std::span<const float> d2_llr,
    std::uint32_t max_iterations,
    bool enable_early_stop,
    bool* crc_ok,
    std::uint32_t* iterations_used) {
#if TURBO_CPP_HAS_CUDA
  return turbo_decode_code_block_gpu_stage1_tile8_experimental_cuda_impl(
      descriptor, d0_llr, d1_llr, d2_llr, max_iterations, enable_early_stop, crc_ok,
      iterations_used);
#else
  (void)descriptor;
  (void)d0_llr;
  (void)d1_llr;
  (void)d2_llr;
  (void)max_iterations;
  (void)enable_early_stop;
  (void)crc_ok;
  (void)iterations_used;
  throw std::runtime_error("GPU Stage1 tile8 experimental requested but CUDA support is not built");
#endif
}

DecodedTransportBlock decode_transport_block(const LteTurboCodecParams& params,
                                             const EncodedTransportBlock& encoded,
                                             DecoderBackend backend) {
  DecodedTransportBlock out;
  out.code_block_crc_ok.reserve(encoded.code_blocks.size());
  out.posterior_llr = std::vector<float>{};

  std::vector<std::uint8_t> tb_bits_with_crc;
  for (const auto& block : encoded.code_blocks) {
    const auto llrs = derate_match_code_block(params, block.descriptor, block.rate_matched_bits);
    const auto d = block.descriptor.d_r;
    bool cb_crc_ok = false;
    std::uint32_t iterations = 0;
    std::vector<float> posterior;
    switch (backend) {
      case DecoderBackend::CpuReference:
        posterior = turbo_decode_code_block_cpu(
            block.descriptor,
            std::span<const float>(llrs.data(), d),
            std::span<const float>(llrs.data() + static_cast<std::ptrdiff_t>(d), d),
            std::span<const float>(llrs.data() + static_cast<std::ptrdiff_t>(2 * d), d),
            params.max_iterations,
            params.enable_early_stop,
            &cb_crc_ok,
            &iterations);
        break;
      case DecoderBackend::GpuStage1Exact:
        posterior = turbo_decode_code_block_gpu_stage1(
            block.descriptor,
            std::span<const float>(llrs.data(), d),
            std::span<const float>(llrs.data() + static_cast<std::ptrdiff_t>(d), d),
            std::span<const float>(llrs.data() + static_cast<std::ptrdiff_t>(2 * d), d),
            params.max_iterations,
            params.enable_early_stop,
            &cb_crc_ok,
            &iterations);
        break;
      case DecoderBackend::GpuStage2Windowed:
        posterior = turbo_decode_code_block_gpu_stage2(
            block.descriptor,
            std::span<const float>(llrs.data(), d),
            std::span<const float>(llrs.data() + static_cast<std::ptrdiff_t>(d), d),
            std::span<const float>(llrs.data() + static_cast<std::ptrdiff_t>(2 * d), d),
            params.max_iterations,
            params.enable_early_stop,
            &cb_crc_ok,
            &iterations);
        break;
    }
    out.iterations_used = std::max(out.iterations_used, iterations);
    auto decoded = hard_decision(posterior);
    decoded.erase(decoded.begin(),
                  decoded.begin() + static_cast<std::ptrdiff_t>(block.descriptor.filler_count));
    if (encoded.code_blocks.size() > 1) {
      decoded.resize(decoded.size() - 24);
    }
    tb_bits_with_crc.insert(tb_bits_with_crc.end(), decoded.begin(), decoded.end());
    out.code_block_crc_ok.push_back(cb_crc_ok || encoded.code_blocks.size() == 1);
    out.posterior_llr->insert(out.posterior_llr->end(), posterior.begin(), posterior.end());
  }
  out.transport_block_crc_ok = check_crc24a(tb_bits_with_crc);
  out.tb_bits.assign(tb_bits_with_crc.begin(),
                     tb_bits_with_crc.end() - static_cast<std::ptrdiff_t>(24));
  return out;
}

std::string to_string(LteChannelType channel_type) {
  switch (channel_type) {
    case LteChannelType::UL_SCH:
      return "UL_SCH";
    case LteChannelType::DL_SCH:
      return "DL_SCH";
    case LteChannelType::PCH:
      return "PCH";
    case LteChannelType::MCH:
      return "MCH";
  }
  throw std::invalid_argument("unknown LTE channel type");
}

}  // namespace turbo_cpp
