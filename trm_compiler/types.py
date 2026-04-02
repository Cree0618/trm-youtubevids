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
        # Use a simple hash-based encoding since benchmarks are categorical
        h = hash(self.benchmark_id) % 256
        return [h / 256.0, float(len(self.benchmark_id)) / 30.0]

    @property
    def label(self) -> str:
        return f"{self.dataset}/{self.benchmark_id}"


@dataclass(frozen=True)
class CompilerFeedback:
    """Feedback after applying a pass."""
    instruction_count: int
    prev_instruction_count: int
    compiled: bool
    reward: float

    def encode(self) -> list[float]:
        """Encode feedback for the model."""
        if self.prev_instruction_count <= 0:
            ratio = 1.0
        else:
            ratio = self.instruction_count / max(self.prev_instruction_count, 1)
        return [
            float(self.instruction_count) / 10000.0,  # normalized
            ratio,
            float(self.compiled),
            self.reward,
        ]

    @staticmethod
    def zero() -> CompilerFeedback:
        return CompilerFeedback(
            instruction_count=0,
            prev_instruction_count=0,
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
        return asdict(self)

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
