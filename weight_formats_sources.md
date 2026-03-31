# Weight Format Sources

Updated: 2026-03-28

This note lists the primary sources used for the Manim explainer comparing low-precision weight and activation formats.

## Core papers and docs

- FP8 Formats for Deep Learning
  - https://arxiv.org/abs/2209.05433
  - Introduces FP8 with E4M3 and E5M2 encodings for deep learning.

- A Study of BFLOAT16 for Deep Learning Training
  - https://arxiv.org/abs/1905.12322
  - Explains why BF16 keeps FP32-like range with fewer mantissa bits.

- NVIDIA Transformer Engine: NVFP4
  - https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html
  - Gives the NVFP4 representation formula and scaling structure.

- Introducing NVFP4 for Efficient and Accurate Low-Precision Inference
  - https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
  - Practical explanation of NVFP4, block size 16, FP8 block scale, FP32 global scale.

- Per-Tensor and Per-Block Scaling Strategies for Effective FP8 Training
  - https://developer.nvidia.com/blog/per-tensor-and-per-block-scaling-strategies-for-effective-fp8-training/
  - Describes MXFP8 block scaling with 32-value blocks and E8M0 scale factors.

## Technical reference points used in the animation

- FP16 / IEEE binary16:
  - 1 sign bit, 5 exponent bits, 10 mantissa bits

- BF16:
  - 1 sign bit, 8 exponent bits, 7 mantissa bits
  - Same exponent width as FP32, therefore much larger dynamic range than FP16

- FP8 E4M3:
  - 1 sign bit, 4 exponent bits, 3 mantissa bits

- MXFP8:
  - FP8 values with block scaling over 32-value blocks
  - Exponent-only E8M0 scale

- NVFP4:
  - E2M1 payload
  - FP8 E4M3 scale per 16-value block
  - FP32 global scale per tensor
