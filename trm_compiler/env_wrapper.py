"""Compiler environment wrapper — supports real CompilerGym and realistic synthetic mode.

On Linux with compiler_gym installed, uses real LLVM. On Windows or without
compiler_gym, uses a realistic synthetic environment that models real pass
ordering dependencies so TRM can learn transferable patterns.
"""
from __future__ import annotations
import math
from typing import Optional

import numpy as np
import torch

from .types import BenchmarkSpec, CompilerFeedback, PassEdit


# Useful LLVM passes for ordering — these are the most impactful passes
USEFUL_PASSES = [
    "mem2reg",
    "simplifycfg",
    "early-cse",
    "instcombine",
    "reassociate",
    "gvn",
    "newgvn",
    "sccp",
    "dce",
    "adce",
    "licm",
    "loop-rotate",
    "indvars",
    "loop-unswitch",
    "loop-deletion",
    "loop-idiom",
    "loop-unroll",
    "loop-vectorize",
    "slp-vectorize",
    "inline",
    "argpromotion",
    "deadargelim",
    "globalopt",
    "globaldce",
    "ipconstprop",
    "ipsccp",
    "prune-eh",
    "strip-dead-prototypes",
    "constmerge",
    "sink",
    "sroa",
    "tailcallelim",
    "correlated-propagation",
    "speculative-execution",
    "jump-threading",
    "simplifycfg-2",
    "instcombine-2",
]

NUM_PASSES = len(USEFUL_PASSES)

# Pass index for quick lookup
_PASS_INDEX = {name: i for i, name in enumerate(USEFUL_PASSES)}


def pass_name_to_id(pass_name: str) -> int:
    return _PASS_INDEX[pass_name]


def pass_id_to_name(pass_id: int) -> str:
    return USEFUL_PASSES[pass_id]


# ──────────────────────────────────────────────────────────────
# CompilerGym action name mapping
# ──────────────────────────────────────────────────────────────

# CompilerGym uses flag-style names like "-mem2reg", "-loop-unroll"
# Some have parameters. This maps our pass names to CompilerGym flags.
COMPILERGYM_FLAG_MAP = {
    "mem2reg": "-mem2reg",
    "simplifycfg": "-simplifycfg",
    "early-cse": "-early-cse",
    "instcombine": "-instcombine",
    "reassociate": "-reassociate",
    "gvn": "-gvn",
    "newgvn": "-newgvn",
    "sccp": "-sccp",
    "dce": "-dce",
    "adce": "-adce",
    "licm": "-licm",
    "loop-rotate": "-loop-rotate",
    "indvars": "-indvars",
    "loop-unswitch": "-loop-unswitch",
    "loop-deletion": "-loop-deletion",
    "loop-idiom": "-loop-idiom",
    "loop-unroll": "-loop-unroll",
    "loop-vectorize": "-loop-vectorize",
    "slp-vectorize": "-slp-vectorize",
    "inline": "-inline",
    "argpromotion": "-argpromotion",
    "deadargelim": "-deadargelim",
    "globalopt": "-globalopt",
    "globaldce": "-globaldce",
    "ipconstprop": "-ipconstprop",
    "ipsccp": "-ipsccp",
    "prune-eh": "-prune-eh",
    "strip-dead-prototypes": "-strip-dead-prototypes",
    "constmerge": "-constmerge",
    "sink": "-sink",
    "sroa": "-sroa",
    "tailcallelim": "-tailcallelim",
    "correlated-propagation": "-correlated-propagation",
    "speculative-execution": "-speculative-execution",
    "jump-threading": "-jump-threading",
    "simplifycfg-2": "-simplifycfg",
    "instcombine-2": "-instcombine",
}


# ──────────────────────────────────────────────────────────────
# Realistic synthetic compiler environment
# ──────────────────────────────────────────────────────────────

# Pass synergy pairs: (predecessor, successor) → bonus multiplier
_PASS_SYNERGIES = {
    ("instcombine", "gvn"): 1.4,
    ("instcombine", "newgvn"): 1.3,
    ("simplifycfg", "instcombine"): 1.3,
    ("instcombine", "simplifycfg"): 1.2,
    ("loop-unroll", "loop-vectorize"): 1.5,
    ("loop-unroll", "slp-vectorize"): 1.3,
    ("loop-rotate", "loop-unroll"): 1.3,
    ("loop-rotate", "licm"): 1.4,
    ("loop-rotate", "loop-vectorize"): 1.3,
    ("mem2reg", "instcombine"): 1.3,
    ("mem2reg", "sroa"): 1.3,
    ("mem2reg", "gvn"): 1.2,
    ("reassociate", "instcombine"): 1.3,
    ("reassociate", "gvn"): 1.2,
    ("gvn", "dce"): 1.3,
    ("gvn", "adce"): 1.2,
    ("gvn", "licm"): 1.2,
    ("inline", "instcombine"): 1.4,
    ("inline", "gvn"): 1.3,
    ("inline", "sroa"): 1.3,
    ("early-cse", "instcombine"): 1.2,
    ("early-cse", "gvn"): 1.2,
    ("indvars", "loop-vectorize"): 1.3,
    ("indvars", "loop-unroll"): 1.2,
}

_ANTI_PATTERNS = {
    ("loop-vectorize", "loop-rotate"): 0.8,
    ("loop-unroll", "loop-rotate"): 0.7,
    ("gvn", "mem2reg"): 0.9,
    ("dce", "instcombine"): 0.9,
}

_PASS_EFFECTIVENESS = {
    "mem2reg":            (0.02, 0.15),
    "simplifycfg":        (0.01, 0.10),
    "early-cse":          (0.01, 0.08),
    "instcombine":        (0.05, 0.25),
    "reassociate":        (0.01, 0.06),
    "gvn":                (0.03, 0.20),
    "newgvn":             (0.03, 0.18),
    "sccp":               (0.01, 0.10),
    "dce":                (0.01, 0.08),
    "adce":               (0.01, 0.06),
    "licm":               (0.02, 0.12),
    "loop-rotate":        (0.00, 0.05),
    "indvars":            (0.00, 0.04),
    "loop-unswitch":      (0.00, 0.06),
    "loop-deletion":      (0.00, 0.08),
    "loop-idiom":         (0.00, 0.06),
    "loop-unroll":        (0.02, 0.15),
    "loop-vectorize":     (0.03, 0.20),
    "slp-vectorize":      (0.02, 0.12),
    "inline":             (0.02, 0.18),
    "argpromotion":       (0.00, 0.04),
    "deadargelim":        (0.00, 0.03),
    "globalopt":          (0.01, 0.08),
    "globaldce":          (0.00, 0.05),
    "ipconstprop":        (0.00, 0.04),
    "ipsccp":             (0.00, 0.05),
    "prune-eh":           (0.00, 0.03),
    "strip-dead-prototypes": (0.00, 0.02),
    "constmerge":         (0.00, 0.02),
    "sink":               (0.00, 0.05),
    "sroa":               (0.02, 0.12),
    "tailcallelim":       (0.00, 0.04),
    "correlated-propagation": (0.00, 0.04),
    "speculative-execution":  (0.00, 0.03),
    "jump-threading":     (0.01, 0.06),
    "simplifycfg-2":      (0.00, 0.05),
    "instcombine-2":      (0.00, 0.08),
}

_BENCHMARK_PROFILES = {
    "qsort":     (True,  True,  False, 0.4, 800),
    "adpcm":     (True,  False, True,  0.6, 1500),
    "blowfish":  (True,  False, False, 0.5, 1200),
    "bzip2":     (True,  False, True,  0.7, 2000),
    "dijkstra":  (True,  True,  False, 0.5, 1000),
    "sha":       (True,  False, True,  0.4, 900),
    "gsm":       (True,  False, True,  0.6, 1400),
    "ispell":    (True,  False, False, 0.5, 1100),
    "jpeg-c":    (True,  False, True,  0.8, 2500),
    "lame":      (True,  False, True,  0.7, 1800),
    "patricia":  (False, True,  False, 0.3, 600),
    "rijndael":  (True,  False, True,  0.6, 1300),
    "stringsearch": (True, True, False, 0.4, 700),
    "susan":     (True,  False, False, 0.5, 1100),
    "tiff2bw":   (True,  False, True,  0.6, 1400),
    "tiff2rgba": (True,  False, True,  0.6, 1500),
    "tiffdither": (True, False, True,  0.5, 1200),
    "tiffmedian": (True, False, True,  0.5, 1100),
}


class SyntheticCompilerEnv:
    """Realistic synthetic compiler environment."""

    def __init__(self, benchmark_id: str = "qsort", seed: int = 42):
        self.benchmark_id = benchmark_id
        self._rng = np.random.RandomState(seed)
        self._observation_dim = 56

        if benchmark_id in _BENCHMARK_PROFILES:
            profile = _BENCHMARK_PROFILES[benchmark_id]
        else:
            profile = (True, False, False, 0.5, 1000)

        self._has_loops, self._has_recursion, self._has_heavy_math, \
            self._complexity, self._base_inst = profile

        self._current_inst = self._base_inst
        self._initial_inst = self._base_inst
        self._applied_passes: list[int] = []
        self._recent_rewards: list[float] = []
        self._cumulative_reward = 0.0
        self._step = 0
        self._done = False
        self._obs = self._generate_observation()

    def reset(self) -> tuple[np.ndarray, int]:
        self._current_inst = self._initial_inst
        self._applied_passes = []
        self._recent_rewards = []
        self._cumulative_reward = 0.0
        self._step = 0
        self._done = False
        self._obs = self._generate_observation()
        return self._obs.copy(), self._initial_inst

    def step(self, pass_id: int) -> tuple[np.ndarray, CompilerFeedback, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")

        pass_name = pass_id_to_name(pass_id)
        prev_inst = self._current_inst

        effectiveness = self._compute_effectiveness(pass_id, pass_name)
        reduction = int(self._current_inst * effectiveness)
        self._current_inst = max(self._current_inst - reduction, 1)

        if prev_inst > 0 and self._current_inst > 0:
            log_reward = math.log(prev_inst / max(self._current_inst, 1))
        else:
            log_reward = 0.0

        self._applied_passes.append(pass_id)
        self._recent_rewards.append(log_reward)
        self._cumulative_reward += log_reward
        self._step += 1

        if self._step >= 40:
            self._done = True
        if log_reward < -1.0:
            self._done = True
        if self._step > 20 and self._rng.rand() < 0.05:
            self._done = True

        self._obs = self._generate_observation()

        feedback = CompilerFeedback(
            instruction_count=self._current_inst,
            prev_instruction_count=prev_inst,
            initial_instruction_count=self._initial_inst,
            compiled=True,
            reward=log_reward,
            cumulative_reward=self._cumulative_reward,
            step=self._step,
            max_steps=40,
            applied_passes=list(self._applied_passes),
            recent_rewards=list(self._recent_rewards),
        )

        info = {
            "pass_name": pass_name,
            "effectiveness": effectiveness,
            "initial_inst_count": self._initial_inst,
            "current_inst_count": self._current_inst,
            "step": self._step,
            "applied_passes": list(self._applied_passes),
        }

        return self._obs.copy(), feedback, self._done, info

    def _compute_effectiveness(self, pass_id: int, pass_name: str) -> float:
        min_pct, max_pct = _PASS_EFFECTIVENESS.get(pass_name, (0.0, 0.05))
        effectiveness = self._rng.uniform(min_pct, max_pct)

        if not self._has_loops and pass_name.startswith("loop-"):
            effectiveness *= 0.1
        if not self._has_heavy_math and pass_name in ("reassociate", "simplifycfg-2"):
            effectiveness *= 0.3
        if not self._has_recursion and pass_name == "tailcallelim":
            effectiveness *= 0.1

        effectiveness *= (0.5 + self._complexity)

        if self._applied_passes:
            last_pass = pass_id_to_name(self._applied_passes[-1])
            key = (last_pass, pass_name)
            if key in _PASS_SYNERGIES:
                effectiveness *= _PASS_SYNERGIES[key]
            for prev_pass_id in self._applied_passes[-3:]:
                prev_name = pass_id_to_name(prev_pass_id)
                key2 = (prev_name, pass_name)
                if key2 in _PASS_SYNERGIES:
                    effectiveness *= (0.5 + 0.5 * _PASS_SYNERGIES[key2])

        if self._applied_passes:
            last_pass = pass_id_to_name(self._applied_passes[-1])
            anti_key = (last_pass, pass_name)
            if anti_key in _ANTI_PATTERNS:
                effectiveness *= _ANTI_PATTERNS[anti_key]

        times_applied = self._applied_passes.count(pass_id)
        if times_applied > 0:
            effectiveness *= (0.5 ** times_applied)

        return effectiveness

    def _generate_observation(self) -> np.ndarray:
        obs = np.zeros(self._observation_dim, dtype=np.float32)
        inst_ratio = self._current_inst / max(self._initial_inst, 1)

        obs[0] = self._has_loops * 10.0 + self._complexity * 5.0
        obs[1] = 5.0 * self._complexity
        obs[2] = 3.0 * self._complexity
        obs[3] = obs[0] - obs[1] - obs[2]
        obs[4] = 4.0 * self._complexity
        obs[5] = 1.0
        obs[6] = obs[4] + obs[5]
        obs[7] = obs[6] / max(obs[0], 1)
        obs[8] = 2.0 * self._has_recursion
        obs[9] = obs[8] * 0.5
        obs[10] = obs[8] * 0.3
        obs[11] = obs[8] * 0.2
        obs[12] = obs[8] / max(obs[0], 1)
        obs[13] = self._current_inst
        obs[14] = 3.0 * self._complexity
        obs[15] = 2.0 * self._complexity
        obs[16] = self._has_heavy_math * 5.0 * self._complexity
        obs[17] = 1.0 * self._has_heavy_math
        obs[18] = 0.5 * self._has_heavy_math
        obs[19] = 8.0 * self._complexity
        obs[20] = 4.0 * self._complexity
        obs[21] = 6.0 * self._complexity * self._has_heavy_math
        obs[22] = 2.0 * self._complexity * self._has_heavy_math
        obs[23] = 3.0 * self._complexity
        obs[24] = 2.0 * self._complexity
        obs[25] = 1.0 * self._complexity
        obs[26] = 4.0 * self._complexity
        obs[27] = 2.0 * self._complexity
        obs[28] = 1.0 * self._complexity
        obs[29] = 6.0 * self._complexity
        obs[30] = 10.0 * self._complexity
        obs[31] = 8.0 * self._complexity
        obs[32] = 2.0 * self._complexity
        obs[33] = 3.0 * self._complexity
        obs[34] = 1.0 * self._complexity
        obs[35] = 5.0 * self._complexity
        obs[36] = 2.0 * self._complexity * self._has_heavy_math
        obs[37] = 1.0 * self._complexity
        obs[38] = obs[35] + obs[36]
        obs[39] = obs[38] / max(obs[13], 1)
        obs[40] = float(self._has_loops) * 5.0 * self._complexity
        obs[41] = obs[40] * 0.8
        obs[42] = obs[40] * 0.3
        obs[43] = obs[40] * 0.1
        obs[44] = obs[40] * 0.05
        obs[45] = obs[44] / max(obs[40], 1)
        obs[46] = obs[43] / max(obs[0], 1)
        obs[47] = 2.0 * self._has_loops
        obs[48] = inst_ratio
        obs[49] = len(self._applied_passes)
        obs[50] = self._has_heavy_math * 1.0
        obs[51] = self._has_recursion * 1.0
        obs[52] = self._has_loops * 1.0
        obs[53] = self._complexity
        obs[54] = inst_ratio * self._complexity
        obs[55] = len(set(self._applied_passes)) / max(NUM_PASSES, 1)

        return obs

    def get_observation_dim(self) -> int:
        return self._observation_dim

    def get_num_passes(self) -> int:
        return NUM_PASSES

    @property
    def initial_inst_count(self) -> int:
        return self._initial_inst

    @property
    def current_inst_count(self) -> int:
        return self._current_inst


# ──────────────────────────────────────────────────────────────
# Real CompilerGym wrapper (Linux only)
# ──────────────────────────────────────────────────────────────


def _has_compilergym() -> bool:
    try:
        import compiler_gym
        return True
    except ImportError:
        return False


class CompilerGymWrapper:
    """Wrapper around CompilerGym's LLVM environment (Linux only).

    Maps our USEFUL_PASSES to CompilerGym's action space and provides
    the same interface as SyntheticCompilerEnv.
    """

    def __init__(
        self,
        benchmark_id: str = "cbench-v1/qsort",
        observation_space: str = "Autophase",
        reward_space: str = "IrInstructionCountOz",
    ):
        self.benchmark_id = benchmark_id
        self.observation_space_name = observation_space
        self.reward_space_name = reward_space
        self.env = None
        self._initial_inst_count = 0
        self._current_inst_count = 0
        self._applied_passes: list[int] = []
        self._recent_rewards: list[float] = []
        self._cumulative_reward = 0.0
        self._step = 0
        self._done = False
        self._action_map: dict[int, int] = {}  # our pass_id -> CompilerGym action

    def open(self):
        try:
            import compiler_gym
        except ImportError:
            raise ImportError(
                "compiler_gym is required on Linux. "
                "Install with: pip install compiler_gym"
            )

        self.env = compiler_gym.make(
            "llvm-v0",
            benchmark=self.benchmark_id,
            observation_space=self.observation_space_name,
            reward_space=self.reward_space_name,
        )

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def _build_action_map(self):
        """Map our pass IDs to CompilerGym action indices.

        CompilerGym's action_space.names contains flag-style names like "-mem2reg".
        We need to find which action index corresponds to each of our passes.
        """
        if self.env is None:
            return

        # Get CompilerGym's action names
        action_names = list(self.env.action_space.names)

        self._action_map = {}
        for pass_id, pass_name in enumerate(USEFUL_PASSES):
            flag = COMPILERGYM_FLAG_MAP.get(pass_name, f"-{pass_name}")
            # Handle duplicate flags (simplifycfg-2, instcombine-2)
            # These map to the same flag but we track them as different passes
            try:
                action_idx = action_names.index(flag)
                self._action_map[pass_id] = action_idx
            except ValueError:
                # Flag not found in CompilerGym — skip this pass
                pass

    def reset(self) -> tuple[np.ndarray, int]:
        obs = self.env.reset()
        self._build_action_map()

        self._initial_inst_count = int(self.env.observation["IrInstructionCount"])
        self._current_inst_count = self._initial_inst_count
        self._applied_passes = []
        self._recent_rewards = []
        self._cumulative_reward = 0.0
        self._step = 0
        self._done = False

        if self.observation_space_name == "Autophase":
            autophase = np.array(
                self.env.observation["Autophase"], dtype=np.float32
            )
        else:
            autophase = np.array(obs, dtype=np.float32)

        return autophase, self._initial_inst_count

    def step(self, pass_id: int) -> tuple[np.ndarray, CompilerFeedback, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")

        pass_name = pass_id_to_name(pass_id)
        prev_inst = self._current_inst_count

        # Map our pass_id to CompilerGym action
        if pass_id not in self._action_map:
            # Pass not available in CompilerGym — return penalty
            feedback = CompilerFeedback(
                instruction_count=self._current_inst_count,
                prev_instruction_count=prev_inst,
                initial_instruction_count=self._initial_inst_count,
                compiled=False,
                reward=-1.0,
                cumulative_reward=self._cumulative_reward,
                step=self._step,
                max_steps=30,
                applied_passes=list(self._applied_passes),
                recent_rewards=list(self._recent_rewards),
            )
            self._done = True
            obs = np.zeros(56, dtype=np.float32)
            if self.observation_space_name == "Autophase":
                obs = np.array(self.env.observation["Autophase"], dtype=np.float32)
            return obs, feedback, True, {"pass_name": pass_name, "error": "unknown_pass"}

        cg_action = self._action_map[pass_id]

        try:
            obs, reward, done, info = self.env.step(cg_action)
            self._current_inst_count = int(self.env.observation["IrInstructionCount"])

            # CompilerGym reward is inst_count reduction (positive = good)
            # Convert to log-ratio like our synthetic env
            if prev_inst > 0 and self._current_inst_count > 0:
                log_reward = math.log(prev_inst / max(self._current_inst_count, 1))
            else:
                log_reward = 0.0

            compiled = True
            self._done = done

        except Exception as e:
            compiled = False
            self._done = True
            log_reward = -1.0
            obs = np.zeros(56, dtype=np.float32)
            if self.observation_space_name == "Autophase":
                obs = np.array(self.env.observation["Autophase"], dtype=np.float32)

        self._applied_passes.append(pass_id)
        self._recent_rewards.append(log_reward)
        self._cumulative_reward += log_reward
        self._step += 1

        if self.observation_space_name == "Autophase":
            autophase = np.array(
                self.env.observation["Autophase"], dtype=np.float32
            )
        else:
            autophase = np.array(obs, dtype=np.float32)

        feedback = CompilerFeedback(
            instruction_count=self._current_inst_count,
            prev_instruction_count=prev_inst,
            initial_instruction_count=self._initial_inst_count,
            compiled=compiled,
            reward=log_reward,
            cumulative_reward=self._cumulative_reward,
            step=self._step,
            max_steps=30,
            applied_passes=list(self._applied_passes),
            recent_rewards=list(self._recent_rewards),
        )

        info = {
            "pass_name": pass_name,
            "initial_inst_count": self._initial_inst_count,
            "current_inst_count": self._current_inst_count,
            "step": self._step,
            "applied_passes": list(self._applied_passes),
            "compiled": compiled,
        }

        return autophase, feedback, self._done, info

    def get_observation_dim(self) -> int:
        return 56

    def get_num_passes(self) -> int:
        return NUM_PASSES

    @property
    def initial_inst_count(self) -> int:
        return self._initial_inst_count

    @property
    def current_inst_count(self) -> int:
        return self._current_inst_count


# ──────────────────────────────────────────────────────────────
# Factory function
# ──────────────────────────────────────────────────────────────


def make_compiler_env(
    benchmark_id: str = "qsort",
    use_compilergym: bool = False,
    seed: int = 42,
):
    """Create a compiler environment.

    Args:
        benchmark_id: Benchmark name (or "cbench-v1/name" for CompilerGym)
        use_compilergym: If True and available, use real CompilerGym
        seed: Random seed

    Returns:
        Environment with reset() and step() methods
    """
    if use_compilergym and _has_compilergym():
        full_id = benchmark_id if "/" in benchmark_id else f"cbench-v1/{benchmark_id}"
        return CompilerGymWrapper(benchmark_id=full_id)
    else:
        return SyntheticCompilerEnv(benchmark_id=benchmark_id, seed=seed)
