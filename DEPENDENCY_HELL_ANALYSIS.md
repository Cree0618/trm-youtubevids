# Dependency Hell Analysis

## Executive Summary

This project suffers from a cascade of dependency conflicts primarily caused by **incompatible Python versions**, **abandoned packages**, and **transitive dependency hell**. The core issues stem from the intersection of three problematic packages: `compiler_gym`, the deprecated `gym` (OpenAI), and `numpy`.

---

## The Root Cause Chain

```
Google Colab Python 3.12
        ↓
   numpy 2.x (pre-installed)
        ↓
   Downgrade to numpy<2.0 required
        ↓
   compiler_gym installation
        ↓
   Requires gym<=0.21 (deprecated)
        ↓
   gym 0.21.0 fails to build on Python 3.12
        ↓
   DEPENDENCY HELL
```

---

## Issue #1: NumPy 2.0 Binary Incompatibility

### Problem
Google Colab now ships with **Python 3.12** and **NumPy 2.0+**. Many packages in this project were developed for NumPy 1.x.

### Evidence
```python
# Error seen:
ValueError: numpy.dtype size changed, may indicate binary incompatibility. 
Expected 96 from C header, got 88 from PyObject
```

### Affected Packages
- `torch` (some versions compiled against NumPy 1.x)
- Various compiled extensions
- Older packages that haven't updated to NumPy 2.0 API

### Current Mitigation
```python
subprocess.run([sys.executable, "-m", "pip", "install", "numpy>=1.26.0,<2.0", "-q"])
```

---

## Issue #2: The Deprecated `gym` Package

### Problem
**`compiler_gym` requires `gym<=0.21,>=0.18.0`** — the deprecated OpenAI gym package (not `gymnasium`).

The `gym` package:
- Hasn't been updated since 2021
- Has broken `setup.py` that fails on Python 3.12
- Requires old versions of `setuptools` and `wheel`

### Root Cause: setuptools/wheel Conflict
The `gym==0.21.0` package has invalid version specifiers in its `setup.py`:

```python
# From gym/setup.py line 20:
opencv-python>=3.  # <- invalid! Missing version number
```

This causes errors with:
- `setuptools>=66.0.0`
- `wheel>=0.40.0`

### Error Message
```
× python setup.py egg_info did not run successfully.
│ exit code: 1
× Encountered error while generating package metadata.
note: This error originates from a subprocess
```

### Stack Overflow Solution
Downgrade `setuptools` and `wheel` before installing `gym`:
```bash
pip install setuptools==65.5.0 "wheel<0.40.0"
```

---

## Issue #3: compiler_gym Not Updated for Python 3.12

### Problem
`compiler_gym` version **0.2.5** (latest, released Nov 2022):
- Was designed for Python 3.7-3.10
- Lists `requires_python = ">=3.7"` (doesn't explicitly forbid 3.12)
- But its dependency on `gym` makes it effectively incompatible with Python 3.12

### Dependencies of compiler_gym
```
compiler_gym
├── gym<=0.21,>=0.18.0  ← THE PROBLEM
├── grpcio<1.44.0,>=1.32.0
├── protobuf<4,>=3.19.0
├── absl-py>=0.10.0
├── docker>=4.0.0
├── fasteners>=0.15
├── deprecated>=1.2.12
└── tabulate>=0.8.2
```

### grpcio Conflict
The `grpcio` dependency also often fails to build from source on Python 3.12.

---

## Issue #4: Transitive Dependency Conflicts

### The Conflict Chain
1. `torch` (pre-installed in Colab) → works with NumPy 2.0
2. Project needs NumPy < 2.0 → reinstall numpy
3. `torch` was compiled against NumPy 2.0 → binary incompatibility
4. Need to reinstall torch after numpy downgrade
5. `compiler_gym` brings in `gym` → fails on Python 3.12
6. `gym` needs old setuptools → breaks other installs

### Circular Problem
```
numpy 2.0 → needs numpy<2 → torch breaks
         → reinstall torch → numpy 2.0 comes back
         → repeat forever
```

---

## Issue #5: Google Colab Specific Issues

### Colab's Python Version
As of 2025, Google Colab runs **Python 3.12.x** by default.

### Pre-installed Packages
Colab pre-installs many packages that conflict:
- `torch` (compiled against NumPy 2.0)
- `numpy` 2.x
- `pandas`, `scikit-learn`, etc.

### Runtime Restarts Required
After changing package versions, the Python kernel must be restarted to clear cached `.pyc` files compiled against the old NumPy version.

---

## Current Project Dependencies

### From `pyproject.toml`
```toml
[project]
name = "trm-experiments"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "torch",
  "numpy",
]

[project.optional-dependencies]
env = ["gymnasium>=1.0"]      # Modern gym replacement
compiler = ["compiler_gym"]    # Real LLVM environments
all = ["gymnasium>=1.0", "compiler_gym"]
```

### Actual Usage
The project uses:
- `torch` - ML framework
- `numpy` - numerical computing
- `gymnasium` - RL environments (modern replacement for gym)
- `compiler_gym` - LLVM compiler environments (requires deprecated `gym`)

---

## File Inventory: Where Dependencies Are Listed

| File | Contents |
|------|----------|
| `pyproject.toml` | Core deps: torch, numpy |
| `requirements-colab.txt` | `torch`, `triton` |
| `complete_run.py` | Installs compiler_gym |
| `complete_run.ipynb` | Installs setuptools fix then compiler_gym |
| `README.md` | Documents `pip install compiler_gym` |
| `trm_compiler_real_llvm.py` | Documents `pip install compiler_gym` |

---

## Code References to compiler_gym

### Direct Imports (Optional)
```python
# trm_compiler/env_wrapper.py
def _has_compilergym() -> bool:
    try:
        import compiler_gym
        return True
    except ImportError:
        return False
```

### Usage Pattern
The project has **graceful degradation**:
- If `compiler_gym` available → use real LLVM
- If not available → use synthetic environment

```python
# trm_compiler_real_llvm.py
use_cg = not args.synthetic
if use_cg:
    try:
        import compiler_gym
    except ImportError:
        print("[warn] falling back to --synthetic")
        use_cg = False
```

---

## Solutions Investigated

### Solution 1: Downgrade setuptools/wheel ✅ (Partial Fix)
```bash
pip install setuptools==65.5.0 "wheel<0.40.0"
pip install compiler_gym
```
**Status:** Works but fragile. Future pip updates will break it again.

### Solution 2: Use Python 3.10/3.11 in Colab
Not possible in standard Colab. Can use Colab Pro to select runtime.

### Solution 3: Remove compiler_gym Dependency ✅ (Recommended)
Use only synthetic mode:
```bash
python trm_compiler_real_llvm.py --synthetic
```
**Status:** Works perfectly, no dependency hell.

### Solution 4: Fork and Fix gym
Clone `gym` repo, fix `setup.py`, publish custom wheel.

### Solution 5: Wait for compiler_gym Update
Unlikely - project appears unmaintained (last release Nov 2022).

---

## Recommendations

### For This Project

1. **Primary: Use Synthetic Mode Only**
   - The synthetic environment is realistic and well-tested
   - Zero external dependencies beyond torch/numpy
   - Works on Windows, macOS, Linux, Colab

2. **Document the workaround**
   - Add clear instructions for Colab installation
   - Use the setuptools/wheel downgrade before compiler_gym

3. **Consider alternatives to compiler_gym**
   - `llvmlite` - Python bindings for LLVM
   - Custom wrapper around `clang` + `opt`
   - These would eliminate the gym dependency entirely

### For Users

**Quick Start (Synthetic Mode - No Issues):**
```bash
pip install torch numpy
pip install -e .
python -m trm_compiler.example
```

**With Real LLVM (Requires Workaround):**
```bash
# First, fix the gym issue
pip install setuptools==65.5.0 "wheel<0.40.0"
# Then install compiler_gym
pip install compiler_gym
# Use real LLVM
python -m trm_compiler.example --compilergym
```

---

## Timeline of Events

| Date | Event |
|------|-------|
| 2021-10 | gym 0.21.0 released (last update) |
| 2022-11 | compiler_gym 0.2.5 released (last update) |
| 2024-06 | NumPy 2.0 released |
| 2024-09 | Python 3.12 released |
| 2025-03 | Colab switches to Python 3.12 |
| 2025-03 | Users start reporting gym install failures |

---

## Conclusion

The dependency hell is caused by:

1. **Abandoned package**: `gym` (OpenAI) hasn't been updated since 2021
2. **Stale dependency**: `compiler_gym` still depends on deprecated `gym`
3. **Python version jump**: Colab upgraded to Python 3.12, breaking old packages
4. **Transitive conflicts**: NumPy 2.0, torch, and gym all conflict

**Best solution**: Use synthetic mode, which eliminates all these dependencies and works reliably across all platforms.

---

*Generated: April 2026*
*Author: Dependency Analysis*
