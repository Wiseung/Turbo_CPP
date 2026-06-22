# Turbo_CPP

LTE Turbo codec reference and CUDA implementation aligned to 3GPP TS 36.212.

This repository contains:

- A CPU reference chain for LTE transport block CRC, code block segmentation,
  turbo encoding, rate matching, de-rate matching, and Log-MAP turbo decoding.
- A CUDA Stage 1 exact BCJR implementation.
- A CUDA Stage 2 window-parallel exact BCJR implementation.

The implementation favors BER/BLER correctness over throughput shortcuts.
