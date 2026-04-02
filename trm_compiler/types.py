from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass(frozen=True)
class BenchmarkSpec:
    """Specification of a compiler benchmark program."""
    benchmark_id: str  # e.g., "cbench-v1/qsort"
    dataset: str = "cbench-v1"

    def encode(self) -> list[float]:
        """One-hot-ish encoding for the model."""
        h = hash(self.benchmark_id) % 256
        return [h / 256.0, float(len(self.benchmark_id)) / 30.0]

    @property
    def label(self) -> str:
        return f"{self.dataset}/{self.benchmark_id}"


# Pass categories for feedback encoding
PASS_CATEGORIES = {
    "memory": {
        "mem2reg", "sroa", "globalopt", "globaldce", "sink",
        "strip-dead-prototypes", "constmerge", "deadargelim",
    },
    "loop": {
        "loop-rotate", "indvars", "loop-unswitch", "loop-deletion",
        "loop-idiom", "loop-unroll", "loop-vectorize",
    },
    "cfg": {
        "simplifycfg", "simplifycfg-2", "jump-threading",
        "speculative-execution", "prune-eh",
    },
    "scalar": {
        "instcombine", "instcombine-2", "early-cse", "reassociate",
        "gvn", "newgvn", "sccp", "dce", "adce", "correlated-propagation",
    },
    "vectorize": {
        "slp-vectorize", "loop-vectorize",
    },
    "inline": {
        "inline", "argpromotion", "ipconstprop", "ipsccp",
    },
    "other": {
        "licm", "tailcallelim",
    },
}

# Known synergies: (predecessor, successor) → bonus
KNOWN_SYNERGIES = [
    ("mem2reg", "instcombine"),
    ("instcombine", "gvn"),
    ("loop-unroll", "loop-vectorize"),
    ("mem2reg", "sroa"),
    ("simplifycfg", "instcombine"),
    ("loop-rotate", "licm"),
    ("inline", "instcombine"),
    ("reassociate", "gvn"),
]


@dataclass
class CompilerFeedback:
    """Rich feedback after applying a pass.

    20-dimensional vector encoding:
    [0]  inst_count_norm      - normalized instruction count (inst / 10000)
    [1]  inst_ratio           - inst / prev_inst (1.0 = no change)
    [2]  compiled             - 1.0 if pass succeeded
    [3]  step_reward          - log-ratio reward for this step
    [4]  cumulative_reward    - total reward so far
    [5]  reduction_pct        - 1 - inst/initial_inst
    [6]  step_norm            - step / max_steps
    [7]  pass_repeated        - 1.0 if this pass was applied before
    [8]  trend_1              - reward of last pass
    [9]  trend_2              - reward 2 passes ago
    [10] trend_3              - reward 3 passes ago
    [11] cat_memory           - fraction of memory passes applied
    [12] cat_loop             - fraction of loop passes applied
    [13] cat_cfg              - fraction of cfg passes applied
    [14] cat_scalar           - fraction of scalar passes applied
    [15] synergy_active       - 1.0 if last 2 passes form a known synergy
    [16] diversity            - unique_passes / total_passes
    [17] diminishing          - reward_this / max(recent_rewards) < 0.5
    [18] velocity             - reward_trend (positive = improving)
    [19] halt_signal          - 1.0 if consecutive zero-reward passes > 3
    """
    instruction_count: int
    prev_instruction_count: int
    initial_instruction_count: int
    compiled: bool
    reward: float
    cumulative_reward: float = 0.0
    step: int = 0
    max_steps: int = 30
    applied_passes: list = field(default_factory=list)
    recent_rewards: list = field(default_factory=list)

    def encode(self) -> list[float]:
        """Encode 20-dim rich feedback vector for the model."""
        inst = self.instruction_count
        prev_inst = self.prev_instruction_count
        init_inst = self.initial_instruction_count

        # [0] Normalized instruction count
        inst_count_norm = float(inst) / 10000.0

        # [1] Instruction ratio
        if prev_inst > 0:
            inst_ratio = inst / prev_inst
        else:
            inst_ratio = 1.0

        # [2] Compiled
        compiled = float(self.compiled)

        # [3] Step reward
        step_reward = self.reward

        # [4] Cumulative reward
        cumulative_reward = self.cumulative_reward

        # [5] Reduction percentage
        if init_inst > 0:
            reduction_pct = 1.0 - (inst / init_inst)
        else:
            reduction_pct = 0.0

        # [6] Step normalized
        step_norm = self.step / max(self.max_steps, 1)

        # [7] Pass repeated
        current_pass = self.applied_passes[-1] if self.applied_passes else -1
        pass_repeated = float(
            current_pass >= 0 and self.applied_passes[:-1].count(current_pass) > 0
        )

        # [8-10] Trend (last 3 rewards, reversed so most recent is trend_1)
        recent = list(reversed(self.recent_rewards[-3:]))
        trend_1 = recent[0] if len(recent) > 0 else 0.0
        trend_2 = recent[1] if len(recent) > 1 else 0.0
        trend_3 = recent[2] if len(recent) > 2 else 0.0

        # [11-14] Pass category counts
        total_passes = max(len(self.applied_passes), 1)
        counts = {cat: 0 for cat in PASS_CATEGORIES}
        for p in self.applied_passes:
            # p is a pass_id (int), need to check pass names
            # We'll pass the pass name separately or encode by category
            pass_name = self._pass_id_to_name(p) if isinstance(p, int) else str(p)
            for cat, members in PASS_CATEGORIES.items():
                if pass_name in members:
                    counts[cat] += 1

        cat_memory = counts["memory"] / total_passes
        cat_loop = counts["loop"] / total_passes
        cat_cfg = counts["cfg"] / total_passes
        cat_scalar = counts["scalar"] / total_passes

        # [15] Synergy active
        synergy_active = 0.0
        if len(self.applied_passes) >= 2:
            prev_name = self._pass_id_to_name(self.applied_passes[-2])
            curr_name = self._pass_id_to_name(self.applied_passes[-1])
            for s_prev, s_curr in KNOWN_SYNERGIES:
                if prev_name == s_prev and curr_name == s_curr:
                    synergy_active = 1.0
                    break

        # [16] Diversity
        unique_passes = len(set(self.applied_passes))
        diversity = unique_passes / total_passes

        # [17] Diminishing returns
        if self.recent_rewards:
            max_recent = max(abs(r) for r in self.recent_rewards[-5:])
            diminishing = float(abs(self.reward) < 0.5 * max_recent) if max_recent > 0 else 0.0
        else:
            diminishing = 0.0

        # [18] Velocity (reward trend direction)
        if len(self.recent_rewards) >= 3:
            velocity = self.recent_rewards[-1] - self.recent_rewards[-3]
        else:
            velocity = 0.0

        # [19] Halt signal (consecutive zero-reward passes)
        consecutive_zeros = 0
        for r in reversed(self.recent_rewards):
            if abs(r) < 1e-6:
                consecutive_zeros += 1
            else:
                break
        halt_signal = float(consecutive_zeros > 3)

        return [
            inst_count_norm,
            inst_ratio,
            compiled,
            step_reward,
            cumulative_reward,
            reduction_pct,
            step_norm,
            pass_repeated,
            trend_1,
            trend_2,
            trend_3,
            cat_memory,
            cat_loop,
            cat_cfg,
            cat_scalar,
            synergy_active,
            diversity,
            diminishing,
            velocity,
            halt_signal,
        ]

    def simple_encode(self) -> list[float]:
        """Encode simple 4-dim feedback (inst_count, ratio, compiled, reward)."""
        if self.prev_instruction_count <= 0:
            ratio = 1.0
        else:
            ratio = self.instruction_count / max(self.prev_instruction_count, 1)
        return [
            float(self.instruction_count) / 10000.0,
            ratio,
            float(self.compiled),
            self.reward,
        ]

    @staticmethod
    def _pass_id_to_name(pass_id: int) -> str:
        """Convert pass_id to name. Imported lazily to avoid circular imports."""
        try:
            from .env_wrapper import pass_id_to_name
            return pass_id_to_name(pass_id)
        except (ImportError, IndexError):
            return f"pass_{pass_id}"

    @staticmethod
    def zero() -> CompilerFeedback:
        return CompilerFeedback(
            instruction_count=0,
            prev_instruction_count=0,
            initial_instruction_count=0,
            compiled=True,
            reward=0.0,
        )


@dataclass(frozen=True)
class PassEdit:
    """A single compiler pass edit."""
    pass_id: int  # index into the pass list
    pass_name: str = ""


@dataclass
class CompilerTraceRecord:
    """One record in a training trace."""
    benchmark: BenchmarkSpec
    observation: list[float]  # Autophase features
    prev_pass_sequence: list[int]  # pass IDs applied so far
    edit: PassEdit  # the edit that was applied
    feedback: CompilerFeedback
    next_observation: list[float]  # after applying edit
    reward: float
    step: int
    score: float = 0.0  # cumulative reward

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CompilerTraceRecord:
        return cls(
            benchmark=BenchmarkSpec(**d["benchmark"]),
            observation=d["observation"],
            prev_pass_sequence=d["prev_pass_sequence"],
            edit=PassEdit(**d["edit"]),
            feedback=CompilerFeedback(**d["feedback"]),
            next_observation=d["next_observation"],
            reward=d["reward"],
            step=d["step"],
            score=d.get("score", 0.0),
        )
