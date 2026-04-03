# TRM Compiler Pipeline - Current Approach

## Latest: Direct LLVM Integration

Instead of CompilerGym dependency hell, use direct LLVM calls:
- `opt` to apply optimization passes
- Get real instruction counts from compiled output
- No service binary required

## How It Works
1. Environment has LLVM/Clang installed (system or conda)
2. `CompilerEnv` uses subprocess to call `opt` with passes
3. Real feedback from actual compilation

## Setup Requirements
- LLVM/Clang installed: `conda install -c conda-forge clang=21`
- Or use MLIR-RL Docker container for pre-built environment

## Files
- `complete_run.py` - main training script  
- `TRM_Colab.ipynb` - Colab notebook
- `trm_compiler_real_llvm.py` - TRM model and environment

## Legacy (DEPRECATED)
CompilerGym approach caused:
- pydantic v2 regex deprecated error
- LLVM service binary missing (returncode 127)
- Colab environment issues
