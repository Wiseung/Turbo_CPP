#include "turbo_cpp/lte_turbo.h"

#include <chrono>
#include <iostream>
#include <sstream>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct BenchCase {
  const char* name;
  std::size_t tb_bits;
  int repeats;
};

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
                         int repeats,
                         int warmup_repeats = 0) {
  for (int i = 0; i < warmup_repeats; ++i) {
    auto decoded = turbo_cpp::decode_transport_block(params, encoded, backend);
    if (!decoded.transport_block_crc_ok) {
      throw std::runtime_error("bench warmup decode CRC failed");
    }
  }
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

void warm_up_gpu(const turbo_cpp::LteTurboCodecParams& params,
                 const turbo_cpp::EncodedTransportBlock& encoded,
                 int warmup_decode_count) {
#if TURBO_CPP_HAS_CUDA
  for (int i = 0; i < warmup_decode_count; ++i) {
    auto decoded = turbo_cpp::decode_transport_block(
        params, encoded, turbo_cpp::DecoderBackend::GpuStage2Windowed);
    if (!decoded.transport_block_crc_ok) {
      throw std::runtime_error("global GPU warmup decode CRC failed");
    }
  }
#else
  (void)params;
  (void)encoded;
  (void)warmup_decode_count;
#endif
}

std::size_t compute_full_rate_g_total(const turbo_cpp::SegmentationResult& segmentation) {
  std::size_t total = 0;
  for (const auto& block : segmentation.code_blocks) {
    total += 3 * (block.size() + 4);
  }
  return total;
}

std::string join_descriptor_field(const turbo_cpp::EncodedTransportBlock& encoded,
                                  std::size_t turbo_cpp::CodeBlockDescriptor::*field) {
  std::ostringstream oss;
  for (std::size_t i = 0; i < encoded.code_blocks.size(); ++i) {
    if (i != 0) {
      oss << ',';
    }
    oss << encoded.code_blocks[i].descriptor.*field;
  }
  return oss.str();
}

template <typename MeasureFn>
void print_backend_result(const BenchCase& bench_case,
                          std::string_view backend,
                          MeasureFn&& measure) {
  try {
    const double ms = measure();
    std::cout << "tb_bits=" << bench_case.tb_bits
              << " backend=" << backend
              << " ms=" << ms << '\n';
  } catch (const std::exception& ex) {
    std::cout << "tb_bits=" << bench_case.tb_bits
              << " backend=" << backend
              << " error=" << ex.what() << '\n';
  }
}

void print_stage2_tb_validation(const BenchCase& bench_case,
                                const turbo_cpp::LteTurboCodecParams& params,
                                const turbo_cpp::EncodedTransportBlock& encoded) {
  try {
    const auto decoded = turbo_cpp::decode_transport_block(
        params, encoded, turbo_cpp::DecoderBackend::GpuStage2Windowed);

    std::size_t tb_bit_disagreements = 0;
    for (std::size_t i = 0; i < decoded.tb_bits.size() && i < encoded.transport_bits.size(); ++i) {
      if (decoded.tb_bits[i] != encoded.transport_bits[i]) {
        ++tb_bit_disagreements;
      }
    }

    std::size_t cb_crc_ok_count = 0;
    for (const bool ok : decoded.code_block_crc_ok) {
      if (ok) {
        ++cb_crc_ok_count;
      }
    }

    std::cout << "tb_bits=" << bench_case.tb_bits
              << " backend=gpu_stage2_tb_check"
              << " transport_crc=" << decoded.transport_block_crc_ok
              << " cb_crc_ok=" << cb_crc_ok_count << '/' << decoded.code_block_crc_ok.size()
              << " tb_bit_disagreements=" << tb_bit_disagreements
              << '\n';
  } catch (const std::exception& ex) {
    std::cout << "tb_bits=" << bench_case.tb_bits
              << " backend=gpu_stage2_tb_check"
              << " error=" << ex.what() << '\n';
  }
}

double measure_tile8_experimental_ms(const turbo_cpp::LteTurboCodecParams& params,
                                     const turbo_cpp::EncodedTransportBlock& encoded,
                                     int repeats,
                                     int warmup_repeats = 0) {
  const auto& block = encoded.code_blocks.front();
  const auto llrs =
      turbo_cpp::derate_match_code_block(params, block.descriptor, block.rate_matched_bits);
  for (int i = 0; i < warmup_repeats; ++i) {
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
      throw std::runtime_error("tile8 experimental warmup posterior empty");
    }
  }
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

void run_case(const BenchCase& bench_case) {
  turbo_cpp::LteTurboCodecParams params;
  params.channel_type = turbo_cpp::LteChannelType::DL_SCH;
  params.transport_block_bits = bench_case.tb_bits;
  params.rv = 0;
  params.num_layers = 1;
  params.mod_order = 2;
  params.max_iterations = 6;
  params.enable_early_stop = false;

  const auto bits = make_bits(bench_case.tb_bits);
  const auto tb_bits_with_crc = turbo_cpp::append_crc24a(bits);
  const auto segmentation = turbo_cpp::segment_transport_block(tb_bits_with_crc);
  params.g_total = compute_full_rate_g_total(segmentation);
  const auto encoded = turbo_cpp::encode_transport_block(params, bits);
  const int cpu_warmup = 1;
  const int gpu_warmup = 2;
  const int global_gpu_warmup = (bench_case.tb_bits >= 6112) ? 3 : 1;

  std::cout << "case=" << bench_case.name
            << " tb_bits=" << bench_case.tb_bits
            << " repeats=" << bench_case.repeats
            << " cb_count=" << encoded.code_blocks.size()
            << " g_total=" << params.g_total
            << " cb_k=" << join_descriptor_field(encoded, &turbo_cpp::CodeBlockDescriptor::k_r)
            << " cb_e=" << join_descriptor_field(encoded, &turbo_cpp::CodeBlockDescriptor::e_r)
            << " cb_filler=" << join_descriptor_field(encoded, &turbo_cpp::CodeBlockDescriptor::filler_count)
            << '\n';

  print_backend_result(bench_case, "cpu", [&] {
    return measure_decode_ms(params, encoded, turbo_cpp::DecoderBackend::CpuReference,
                             bench_case.repeats, cpu_warmup);
  });

#if TURBO_CPP_HAS_CUDA
  warm_up_gpu(params, encoded, global_gpu_warmup);

  print_backend_result(bench_case, "gpu_stage1", [&] {
    return measure_decode_ms(params, encoded, turbo_cpp::DecoderBackend::GpuStage1Exact,
                             bench_case.repeats, gpu_warmup);
  });
  print_backend_result(bench_case, "gpu_stage1_tile8_exp", [&] {
    if (encoded.code_blocks.size() != 1) {
      throw std::runtime_error("single-code-block only");
    }
    return measure_tile8_experimental_ms(params, encoded, bench_case.repeats, gpu_warmup);
  });
  print_stage2_tb_validation(bench_case, params, encoded);
  print_backend_result(bench_case, "gpu_stage2", [&] {
    return measure_decode_ms(params, encoded, turbo_cpp::DecoderBackend::GpuStage2Windowed,
                             bench_case.repeats, gpu_warmup);
  });
#endif
}

}  // namespace

int main() {
  try {
    const std::vector<BenchCase> cases = {
        {"micro_single_cb", 40, 20},
        {"small_single_cb", 512, 10},
        {"max_single_cb", 6112, 3},
        {"two_code_blocks", 12000, 2},
    };

    if (const char* filter = std::getenv("TURBO_CPP_BENCH_CASE"); filter != nullptr) {
      for (const auto& bench_case : cases) {
        if (std::string_view(filter) == bench_case.name) {
          run_case(bench_case);
          return 0;
        }
      }
      throw std::runtime_error("unknown TURBO_CPP_BENCH_CASE");
    }

    for (const auto& bench_case : cases) {
      run_case(bench_case);
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "bench failure: " << ex.what() << '\n';
    return 1;
  }
}
