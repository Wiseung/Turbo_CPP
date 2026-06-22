#include "turbo_cpp/lte_turbo.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void test_crc() {
  const std::vector<std::uint8_t> bits = {1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1};
  const auto crc24a = turbo_cpp::append_crc24a(bits);
  const auto crc24b = turbo_cpp::append_crc24b(bits);
  const auto crc16 = turbo_cpp::append_crc16(bits);
  const auto crc8 = turbo_cpp::append_crc8(bits);
  expect(turbo_cpp::check_crc24a(crc24a), "CRC24A failed");
  expect(turbo_cpp::check_crc24b(crc24b), "CRC24B failed");
  expect(turbo_cpp::check_crc16(crc16), "CRC16 failed");
  expect(turbo_cpp::check_crc8(crc8), "CRC8 failed");
}

void test_qpp_table() {
  const auto table = turbo_cpp::qpp_interleaver_table();
  expect(table.size() == 188, "QPP table size mismatch");
  expect(table.front().k == 40 && table.front().f1 == 3 && table.front().f2 == 10,
         "QPP first entry mismatch");
  expect(table.back().k == 6144 && table.back().f1 == 263 && table.back().f2 == 480,
         "QPP last entry mismatch");
}

void test_segmentation() {
  std::vector<std::uint8_t> bits_40(40, 1U);
  const auto seg_40 = turbo_cpp::segment_transport_block(bits_40);
  expect(seg_40.c == 1, "B=40 should stay one block");
  expect(seg_40.code_blocks.front().size() == 40, "B=40 K mismatch");

  std::vector<std::uint8_t> bits_6145(6145, 0U);
  const auto seg_6145 = turbo_cpp::segment_transport_block(bits_6145);
  expect(seg_6145.c > 1, "B=6145 should segment");
  expect(seg_6145.filler_bits > 0, "B=6145 should create filler bits");
}

void test_roundtrip_small() {
  turbo_cpp::LteTurboCodecParams params;
  params.channel_type = turbo_cpp::LteChannelType::DL_SCH;
  params.transport_block_bits = 40;
  params.g_total = 3 * (64 + 4);
  params.rv = 0;
  params.num_layers = 1;
  params.mod_order = 2;
  params.max_iterations = 6;
  params.enable_early_stop = false;

  std::vector<std::uint8_t> bits(40);
  for (std::size_t i = 0; i < bits.size(); ++i) {
    bits[i] = static_cast<std::uint8_t>((i * 7 + 3) & 1U);
  }

  const auto encoded = turbo_cpp::encode_transport_block(params, bits);
  const auto llrs = turbo_cpp::derate_match_code_block(params,
                                                       encoded.code_blocks.front().descriptor,
                                                       encoded.code_blocks.front().rate_matched_bits);
  const auto& block = encoded.code_blocks.front();
  bool llr_mismatch = false;
  for (std::size_t i = 0; i < block.descriptor.k_r; ++i) {
    const int expected = block.systematic[i] ? 1 : -1;
    const int observed = llrs[i] >= 0.0F ? 1 : -1;
    if (expected != observed) {
      llr_mismatch = true;
      break;
    }
  }
  if (llr_mismatch) {
    std::cerr << "Systematic LLR mismatch after de-rate matching\n";
    for (std::size_t i = 0; i < block.descriptor.k_r; ++i) {
      const int expected = block.systematic[i] ? 1 : -1;
      const int observed = llrs[i] >= 0.0F ? 1 : -1;
      if (expected != observed) {
        std::cerr << "first mismatch at " << i << " expected " << expected
                  << " observed " << observed << " llr " << llrs[i] << '\n';
        break;
      }
    }
  }
  bool cb_crc_ok = false;
  std::uint32_t iterations = 0;
  const auto posterior = turbo_cpp::turbo_decode_code_block_cpu(
      block.descriptor,
      std::span<const float>(llrs.data(), block.descriptor.d_r),
      std::span<const float>(llrs.data() + static_cast<std::ptrdiff_t>(block.descriptor.d_r),
                             block.descriptor.d_r),
      std::span<const float>(
          llrs.data() + static_cast<std::ptrdiff_t>(2 * block.descriptor.d_r),
          block.descriptor.d_r),
      params.max_iterations,
      params.enable_early_stop,
      &cb_crc_ok,
      &iterations);
  expect(!posterior.empty(), "Posterior LLRs should not be empty");
  const auto decoded = turbo_cpp::decode_transport_block(params, encoded,
                                                         turbo_cpp::DecoderBackend::CpuReference);
  if (decoded.tb_bits != bits) {
    std::cerr << "Expected bits:\n";
    for (const auto bit : bits) {
      std::cerr << static_cast<int>(bit);
    }
    std::cerr << "\nDecoded bits:\n";
    for (const auto bit : decoded.tb_bits) {
      std::cerr << static_cast<int>(bit);
    }
    std::cerr << "\nTB CRC OK: " << decoded.transport_block_crc_ok
              << " iterations: " << decoded.iterations_used << '\n';
  }
  expect(decoded.tb_bits == bits, "CPU reference roundtrip mismatch");
  expect(decoded.transport_block_crc_ok, "TB CRC should pass after roundtrip");

#if TURBO_CPP_HAS_CUDA
  const auto decoded_stage1 = turbo_cpp::decode_transport_block(
      params, encoded, turbo_cpp::DecoderBackend::GpuStage1Exact);
  expect(decoded_stage1.tb_bits == bits, "GPU Stage1 roundtrip mismatch");
  expect(decoded_stage1.transport_block_crc_ok, "GPU Stage1 CRC should pass");
  expect(decoded_stage1.posterior_llr.has_value(), "GPU Stage1 posterior missing");
  expect(decoded.posterior_llr.has_value(), "CPU posterior missing");
  expect(decoded_stage1.posterior_llr->size() == decoded.posterior_llr->size(),
         "CPU vs GPU Stage1 posterior size mismatch");
  for (std::size_t i = 0; i < decoded.posterior_llr->size(); ++i) {
    if (std::fabs(decoded.posterior_llr->at(i) - decoded_stage1.posterior_llr->at(i)) > 2e-5F) {
      std::cerr << std::setprecision(9)
                << "Stage1 mismatch at " << i
                << " cpu=" << decoded.posterior_llr->at(i)
                << " gpu=" << decoded_stage1.posterior_llr->at(i)
                << " diff=" << std::fabs(decoded.posterior_llr->at(i) - decoded_stage1.posterior_llr->at(i))
                << '\n';
      throw std::runtime_error("CPU vs GPU Stage1 posterior mismatch");
    }
  }

  {
    bool crc_ok_exp = false;
    std::uint32_t iters_exp = 0;
    const auto experimental_posterior = turbo_cpp::turbo_decode_code_block_gpu_stage1_tile8_experimental(
        block.descriptor,
        std::span<const float>(llrs.data(), block.descriptor.d_r),
        std::span<const float>(llrs.data() + static_cast<std::ptrdiff_t>(block.descriptor.d_r),
                               block.descriptor.d_r),
        std::span<const float>(llrs.data() + static_cast<std::ptrdiff_t>(2 * block.descriptor.d_r),
                               block.descriptor.d_r),
        params.max_iterations,
        params.enable_early_stop,
        &crc_ok_exp,
        &iters_exp);
    expect(decoded_stage1.posterior_llr->size() == experimental_posterior.size(),
           "Stage1 vs tile8 experimental size mismatch");
    for (std::size_t i = 0; i < experimental_posterior.size(); ++i) {
      expect(std::fabs(decoded_stage1.posterior_llr->at(i) - experimental_posterior[i]) <= 2e-5F,
             "Stage1 vs tile8 experimental posterior mismatch");
    }
  }

  const auto decoded_stage2 = turbo_cpp::decode_transport_block(
      params, encoded, turbo_cpp::DecoderBackend::GpuStage2Windowed);
  expect(decoded_stage2.tb_bits == bits, "GPU Stage2 roundtrip mismatch");
  expect(decoded_stage2.transport_block_crc_ok, "GPU Stage2 CRC should pass");
  expect(decoded_stage2.posterior_llr.has_value(), "GPU Stage2 posterior missing");
  for (std::size_t i = 0; i < decoded.posterior_llr->size(); ++i) {
    expect(std::fabs(decoded.posterior_llr->at(i) - decoded_stage2.posterior_llr->at(i)) <= 5e-4F,
           "CPU vs GPU Stage2 posterior mismatch");
  }
#endif
}

void test_roundtrip_mult_window() {
  turbo_cpp::LteTurboCodecParams params;
  params.channel_type = turbo_cpp::LteChannelType::DL_SCH;
  params.transport_block_bits = 512;
  params.g_total = 3 * (536 + 4);
  params.rv = 0;
  params.num_layers = 1;
  params.mod_order = 2;
  params.max_iterations = 6;
  params.enable_early_stop = false;

  std::vector<std::uint8_t> bits(512);
  for (std::size_t i = 0; i < bits.size(); ++i) {
    bits[i] = static_cast<std::uint8_t>(((i * 11) + 5) & 1U);
  }

  const auto encoded = turbo_cpp::encode_transport_block(params, bits);
  const auto decoded = turbo_cpp::decode_transport_block(params, encoded,
                                                         turbo_cpp::DecoderBackend::CpuReference);
  expect(decoded.tb_bits == bits, "CPU multi-window roundtrip mismatch");
  expect(decoded.transport_block_crc_ok, "CPU multi-window CRC should pass");

#if TURBO_CPP_HAS_CUDA
  const auto decoded_stage1 = turbo_cpp::decode_transport_block(
      params, encoded, turbo_cpp::DecoderBackend::GpuStage1Exact);
  expect(decoded_stage1.tb_bits == bits, "GPU Stage1 multi-window mismatch");

  {
    const auto llrs_multi = turbo_cpp::derate_match_code_block(
        params, encoded.code_blocks.front().descriptor, encoded.code_blocks.front().rate_matched_bits);
    bool crc_ok_exp = false;
    std::uint32_t iters_exp = 0;
    const auto experimental_posterior = turbo_cpp::turbo_decode_code_block_gpu_stage1_tile8_experimental(
        encoded.code_blocks.front().descriptor,
        std::span<const float>(llrs_multi.data(), encoded.code_blocks.front().descriptor.d_r),
        std::span<const float>(
            llrs_multi.data() + static_cast<std::ptrdiff_t>(encoded.code_blocks.front().descriptor.d_r),
            encoded.code_blocks.front().descriptor.d_r),
        std::span<const float>(
            llrs_multi.data() +
                static_cast<std::ptrdiff_t>(2 * encoded.code_blocks.front().descriptor.d_r),
            encoded.code_blocks.front().descriptor.d_r),
        params.max_iterations,
        params.enable_early_stop,
        &crc_ok_exp,
        &iters_exp);
    expect(decoded_stage1.posterior_llr->size() == experimental_posterior.size(),
           "Stage1 vs tile8 experimental multi-window size mismatch");
    for (std::size_t i = 0; i < experimental_posterior.size(); ++i) {
      expect(std::fabs(decoded_stage1.posterior_llr->at(i) - experimental_posterior[i]) <= 2e-5F,
             "Stage1 vs tile8 experimental multi-window mismatch");
    }
  }

  const auto decoded_stage2 = turbo_cpp::decode_transport_block(
      params, encoded, turbo_cpp::DecoderBackend::GpuStage2Windowed);
  expect(decoded_stage2.tb_bits == bits, "GPU Stage2 multi-window mismatch");
  expect(decoded_stage2.posterior_llr.has_value(), "GPU Stage2 multi-window posterior missing");
  expect(decoded_stage1.posterior_llr.has_value(), "GPU Stage1 multi-window posterior missing");
  for (std::size_t i = 0; i < decoded_stage1.posterior_llr->size(); ++i) {
    expect(std::fabs(decoded_stage1.posterior_llr->at(i) - decoded_stage2.posterior_llr->at(i)) <= 5e-4F,
           "Stage2 vs Stage1 multi-window posterior mismatch");
  }
#endif
}

}  // namespace

int main() {
  try {
    test_crc();
    test_qpp_table();
    test_segmentation();
    test_roundtrip_small();
    test_roundtrip_mult_window();
    std::cout << "All tests passed\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Test failure: " << ex.what() << '\n';
    return 1;
  }
}
