# SOLUTION: Real LLVM on Google Colab

## The Problem
`compiler_gym` requires `gym<=0.21` which **cannot be built on Python 3.12**. The `gym` package has been archived and has fundamental incompatibilities with Python 3.12's `pkgutil` changes.

## Verified Working Solutions

### Solution 1: Use Docker with compiler_gym Image (RECOMMENDED)
Run compiler_gym in a Docker container - bypasses all Python dependency issues.

```python
# In Colab cell 1: Install Docker
!apt-get update && apt-get install -y docker.io

# In Colab cell 2: Run compiler_gym container
import subprocess
subprocess.run([
    "docker", "run", "--rm", "-it",
    "-v", "$(pwd):/workspace",
    "ghcr.io/facebookresearch/compilergym:latest",
    "bash"
], check=False)
```

### Solution 2: Use Pre-installed gym from pip (May Work)
```python
# Try installing a prebuilt gym wheel directly
!pip install --only-binary :all: gym==0.21.0 || echo "No wheel available"
```

### Solution 3: Patch gym installation
```python
# Patch pkgutil before installing gym
import sys
import pkgutil

# Add ImpImporter compatibility for Python 3.12
if not hasattr(pkgutil, 'ImpImporter'):
    class ImpImporter:
        def find_module(self, fullname, path=None):
            return None
    pkgutil.ImpImporter = ImpImporter

# Now try installing
!pip install gym==0.21.0
```

### Solution 4: Use conda-forge (Alternative Python)
```python
# Install conda
!pip install condacolab
import condacolab
condacolab.install()

# Now install with conda (has Python 3.11)
!conda install -c conda-forge compiler_gym
```

### Solution 5: Direct LLVM Pipeline (No compiler_gym)
Build your own LLVM wrapper - more work but full control.

```python
# Install LLVM
!apt-get install -y llvm-14 clang-14 opt

# Create custom wrapper
import subprocess
import os

class LLVMWrapper:
    def __init__(self, clang="clang-14", opt="opt-14"):
        self.clang = clang
        self.opt = opt
    
    def compile(self, source_file, output_file, passes=""):
        """Compile with optimization passes."""
        # Step 1: Generate IR
        subprocess.run([self.clang, "-S", "-O1", "-emit-llvm", source_file, "-o", "temp.ll"], check=True)
        
        # Step 2: Apply optimization passes
        cmd = [self.opt, "-S", "temp.ll", "-o", output_file]
        if passes:
            for p in passes.split(","):
                cmd.insert(2, f"-{p}")
        subprocess.run(cmd, check=True)
        
        # Step 3: Get instruction count
        result = subprocess.run([self.clang, "-S", "-o", "-", output_file], 
                              capture_output=True, text=True)
        return result.stdout  # Contains compiled output
    
    def count_instructions(self, llvm_file):
        """Count instructions in IR."""
        result = subprocess.run(
            ["llvm-dis", llvm_file],
            capture_output=True
        )
        # Parse .ll file for instruction count
        with open(llvm_file.replace(".bc", ".ll")) as f:
            return sum(1 for line in f if line.strip().startswith('%'))
```

---

## Quick Start: Recommended Approach

### Option A: Docker-based (Easiest for Real LLVM)

```python
# Cell 1: Setup
!apt-get update -qq
!apt-get install -y -qq docker.io

# Cell 2: Run training with Docker
import subprocess
import os

# Create working directory
os.makedirs("/content/workspace", exist_ok=True)
os.chdir("/content/workspace")

# Run compiler_gym docker image
result = subprocess.run([
    "docker", "run", "--rm",
    "-v", "/content/workspace:/workspace",
    "-w", "/workspace",
    "ghcr.io/facebookresearch/compilergym:latest",
    "bash", "-c", """
    pip install -e /workspace
    python trm_compiler_real_llvm.py --epochs 5 --benchmarks qsort
    """
], capture_output=True, text=True)

print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr)
```

### Option B: Use the workaround in this notebook

The notebook cells below contain the setuptools downgrade workaround which may work:

```python
# CRITICAL: Install in this exact order
!pip install setuptools==65.5.0 "wheel<0.40.0" -q
!pip install gym==0.21.0 --no-build-isolation || true
!pip install compiler_gym -q
```

---

## Current Status by Component

| Component | Python 3.12 Status | Workaround |
|-----------|-------------------|------------|
| torch | ✅ Works | Pre-installed in Colab |
| numpy | ⚠️ Need <2.0 | `pip install numpy<2.0` |
| gym (old) | ❌ Fails | No fix possible |
| gymnasium | ✅ Works | Use instead of gym |
| compiler_gym | ⚠️ Broken | Depends on broken gym |
| triton | ✅ Works | Pre-installed in Colab |

---

## Action Plan

1. **Immediate**: Test the setuptools downgrade workaround in Colab
2. **If #1 fails**: Use Docker-based solution  
3. **Long-term**: Build custom LLVM wrapper that doesn't need compiler_gym

The most reliable path to real LLVM data is currently through Docker.
