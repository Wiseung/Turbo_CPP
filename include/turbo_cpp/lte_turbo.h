#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace turbo_cpp {

enum class LteChannelType {
  UL_SCH,
  DL_SCH,
  PCH,
  MCH,
};

enum class DecoderBackend {
  CpuReference,
  GpuStage1Exact,
  GpuStage2Windowed,
};

struct LteTurboCodecParams {
  LteChannelType channel_type = LteChannelType::DL_SCH;
  std::size_t transport_block_bits = 0;
  std::size_t g_total = 0;
  std::uint32_t rv = 0;
  std::uint32_t num_layers = 1;
  std::uint32_t mod_order = 2;
  std::size_t n_soft = 0;
  std::uint32_t max_iterations = 8;
  bool enable_early_stop = false;
};

struct CodeBlockDescriptor {
  std::size_t block_index = 0;
  std::size_t k_r = 0;
  std::size_t d_r = 0;
  std::size_t e_r = 0;
  std::size_t filler_count = 0;
  std::size_t qpp_index = 0;
};

struct EncodedCodeBlock {
  CodeBlockDescriptor descriptor;
  std::vector<std::uint8_t> cb_bits;
  std::vector<std::uint8_t> systematic;
  std::vector<std::uint8_t> parity0;
  std::vector<std::uint8_t> parity1;
  std::array<std::uint8_t, 3> upper_tail_systematic{};
  std::array<std::uint8_t, 3> upper_tail_parity{};
  std::array<std::uint8_t, 3> lower_tail_systematic{};
  std::array<std::uint8_t, 3> lower_tail_parity{};
  std::vector<std::uint8_t> rate_matched_bits;
};

struct EncodedTransportBlock {
  LteTurboCodecParams params;
  std::vector<std::uint8_t> tb_bits_with_crc;
  std::vector<EncodedCodeBlock> code_blocks;
  std::vector<std::uint8_t> transport_bits;
};

struct DecodedTransportBlock {
  std::vector<std::uint8_t> tb_bits;
  std::vector<bool> code_block_crc_ok;
  bool transport_block_crc_ok = false;
  std::uint32_t iterations_used = 0;
  std::optional<std::vector<float>> posterior_llr;
};

struct QppEntry {
  std::size_t k = 0;
  std::size_t f1 = 0;
  std::size_t f2 = 0;
};

struct SegmentationResult {
  std::size_t c = 0;
  std::size_t c_plus = 0;
  std::size_t c_minus = 0;
  std::size_t k_plus = 0;
  std::size_t k_minus = 0;
  std::size_t filler_bits = 0;
  std::size_t l = 0;
  std::vector<std::vector<std::uint8_t>> code_blocks;
};

std::span<const QppEntry> qpp_interleaver_table();
const QppEntry& qpp_entry_for_k(std::size_t k);

std::vector<std::uint8_t> append_crc24a(std::span<const std::uint8_t> bits);
std::vector<std::uint8_t> append_crc24b(std::span<const std::uint8_t> bits);
std::vector<std::uint8_t> append_crc16(std::span<const std::uint8_t> bits);
std::vector<std::uint8_t> append_crc8(std::span<const std::uint8_t> bits);
bool check_crc24a(std::span<const std::uint8_t> bits_with_crc);
bool check_crc24b(std::span<const std::uint8_t> bits_with_crc);
bool check_crc16(std::span<const std::uint8_t> bits_with_crc);
bool check_crc8(std::span<const std::uint8_t> bits_with_crc);

SegmentationResult segment_transport_block(std::span<const std::uint8_t> bits_with_crc);

std::vector<std::size_t> qpp_permutation(std::size_t k);
std::vector<std::size_t> qpp_inverse_permutation(std::size_t k);

EncodedCodeBlock turbo_encode_code_block(const CodeBlockDescriptor& descriptor,
                                         std::span<const std::uint8_t> cb_bits);
EncodedTransportBlock encode_transport_block(const LteTurboCodecParams& params,
                                             std::span<const std::uint8_t> transport_bits);

std::vector<float> derate_match_code_block(const LteTurboCodecParams& params,
                                           const CodeBlockDescriptor& descriptor,
                                           std::span<const std::uint8_t> rate_matched_bits);

std::vector<float> turbo_decode_code_block_cpu(const CodeBlockDescriptor& descriptor,
                                               std::span<const float> d0_llr,
                                               std::span<const float> d1_llr,
                                               std::span<const float> d2_llr,
                                               std::uint32_t max_iterations,
                                               bool enable_early_stop,
                                               bool* crc_ok = nullptr,
                                               std::uint32_t* iterations_used = nullptr);

std::vector<float> turbo_decode_code_block_gpu_stage1(const CodeBlockDescriptor& descriptor,
                                                      std::span<const float> d0_llr,
                                                      std::span<const float> d1_llr,
                                                      std::span<const float> d2_llr,
                                                      std::uint32_t max_iterations,
                                                      bool enable_early_stop,
                                                      bool* crc_ok = nullptr,
                                                      std::uint32_t* iterations_used = nullptr);

std::vector<float> turbo_decode_code_block_gpu_stage2(const CodeBlockDescriptor& descriptor,
                                                      std::span<const float> d0_llr,
                                                      std::span<const float> d1_llr,
                                                      std::span<const float> d2_llr,
                                                      std::uint32_t max_iterations,
                                                      bool enable_early_stop,
                                                      bool* crc_ok = nullptr,
                                                      std::uint32_t* iterations_used = nullptr);

std::vector<float> turbo_decode_code_block_gpu_stage1_tile8_experimental(
    const CodeBlockDescriptor& descriptor,
    std::span<const float> d0_llr,
    std::span<const float> d1_llr,
    std::span<const float> d2_llr,
    std::uint32_t max_iterations,
    bool enable_early_stop,
    bool* crc_ok = nullptr,
    std::uint32_t* iterations_used = nullptr);

DecodedTransportBlock decode_transport_block(const LteTurboCodecParams& params,
                                             const EncodedTransportBlock& encoded,
                                             DecoderBackend backend);

std::string to_string(LteChannelType channel_type);

}  // namespace turbo_cpp
