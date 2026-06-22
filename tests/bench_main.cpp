#include "turbo_cpp/lte_turbo.h"

#include <chrono>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::vector<std::uint8_t> make_bits(std::size_t n) {
  std::vector<std::uint8_t> bits(n);
  for (std::size_t i = 0; i < n; ++i) {
    bits[i] = static_cast<std::uint8_t>(((i * 13) + 7) & 1U);
  }
  return bits;
}

double measure_decode_ms(const turbo_cpp::LteTurboCodecParams& params,
                         const turbo_cpp::EncodedTransportBlock& encoded,
                         turbo_cpp::DecoderBackend backend,
                         int repeats) {
  const auto start = Clock::now();
  for (int i = 0; i < repeats; ++i) {
    auto decoded = turbo_cpp::decode_transport_block(params, encoded, backend);
    if (!decoded.transport_block_crc_ok) {
      throw std::runtime_error("bench decode CRC failed");
    }
  }
  const auto end = Clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count() / repeats;
}

double measure_tile8_experimental_ms(const turbo_cpp::LteTurboCodecParams& params,
                                     const turbo_cpp::EncodedTransportBlock& encoded,
                                     int repeats) {
  const auto& block = encoded.code_blocks.front();
  const auto llrs =
      turbo_cpp::derate_match_code_block(params, block.descriptor, block.rate_matched_bits);
  const auto start = Clock::now();
  for (int i = 0; i < repeats; ++i) {
    bool crc_ok = false;
    std::uint32_t iters = 0;
    const auto posterior = turbo_cpp::turbo_decode_code_block_gpu_stage1_tile8_experimental(
        block.descriptor,
        std::span<const float>(llrs.data(), block.descriptor.d_r),
        std::span<const float>(llrs.data() + static_cast<std::ptrdiff_t>(block.descriptor.d_r),
                               block.descriptor.d_r),
        std::span<const float>(llrs.data() + static_cast<std::ptrdiff_t>(2 * block.descriptor.d_r),
                               block.descriptor.d_r),
        params.max_iterations,
        params.enable_early_stop,
        &crc_ok,
        &iters);
    if (posterior.empty()) {
      throw std::runtime_error("tile8 experimental posterior empty");
    }
  }
  const auto end = Clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count() / repeats;
}

void run_case(std::size_t tb_bits, int repeats) {
  turbo_cpp::LteTurboCodecParams params;
  params.channel_type = turbo_cpp::LteChannelType::DL_SCH;
  params.transport_block_bits = tb_bits;
  params.rv = 0;
  params.num_layers = 1;
  params.mod_order = 2;
  params.max_iterations = 6;
  params.enable_early_stop = false;

  const std::size_t encoded_block_k = tb_bits + 24;
  params.g_total = 3 * (encoded_block_k + 4);

  const auto bits = make_bits(tb_bits);
  const auto encoded = turbo_cpp::encode_transport_block(params, bits);

  const double cpu_ms = measure_decode_ms(params, encoded, turbo_cpp::DecoderBackend::CpuReference, repeats);
  std::cout << "tb_bits=" << tb_bits << " backend=cpu ms=" << cpu_ms << '\n';

#if TURBO_CPP_HAS_CUDA
  const double stage1_ms =
      measure_decode_ms(params, encoded, turbo_cpp::DecoderBackend::GpuStage1Exact, repeats);
  const double stage1_tile8_exp_ms =
      measure_tile8_experimental_ms(params, encoded, repeats);
  const double stage2_ms =
      measure_decode_ms(params, encoded, turbo_cpp::DecoderBackend::GpuStage2Windowed, repeats);
  std::cout << "tb_bits=" << tb_bits << " backend=gpu_stage1 ms=" << stage1_ms << '\n';
  std::cout << "tb_bits=" << tb_bits << " backend=gpu_stage1_tile8_exp ms=" << stage1_tile8_exp_ms << '\n';
  std::cout << "tb_bits=" << tb_bits << " backend=gpu_stage2 ms=" << stage2_ms << '\n';
#endif
}

}  // namespace

int main() {
  try {
    run_case(40, 20);
    run_case(512, 10);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "bench failure: " << ex.what() << '\n';
    return 1;
  }
}
