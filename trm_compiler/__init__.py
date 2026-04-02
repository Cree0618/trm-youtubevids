from .types import (
    BenchmarkSpec,
    CompilerFeedback,
    CompilerTraceRecord,
    PassEdit,
)
from .env_wrapper import (
    CompilerGymWrapper,
    SyntheticCompilerEnv,
    make_compiler_env,
    NUM_PASSES,
    pass_id_to_name,
    pass_name_to_id,
)
from .model import TinyPassOrderingRefiner, rollout_pass_optimizer, rollout_blind, rollout_closed_loop
from .data import generate_compiler_traces, CompilerTraceDataset
from .training import compute_compiler_losses
from .baselines import (
    random_search,
    greedy_search,
    beam_search,
    run_optimization_level,
    run_full_benchmark,
)
from .eval_real_llvm import (
    eval_on_compilergym,
    run_pipeline_on_compilergym,
    evaluate_model_on_real_llvm,
    print_results_table,
    random_selector,
    trm_selector,
)

try:
    from .real_llvm_env import RealLLVMEnv, AVAILABLE_BENCHMARKS
    HAS_REAL_LLVM = True
except ImportError:
    HAS_REAL_LLVM = False
    RealLLVMEnv = None
    AVAILABLE_BENCHMARKS = []

__all__ = [
    "BenchmarkSpec",
    "CompilerFeedback",
    "CompilerTraceRecord",
    "PassEdit",
    "CompilerGymWrapper",
    "SyntheticCompilerEnv",
    "make_compiler_env",
    "NUM_PASSES",
    "pass_id_to_name",
    "pass_name_to_id",
    "TinyPassOrderingRefiner",
    "rollout_pass_optimizer",
    "rollout_blind",
    "rollout_closed_loop",
    "generate_compiler_traces",
    "CompilerTraceDataset",
    "compute_compiler_losses",
    "random_search",
    "greedy_search",
    "beam_search",
    "run_optimization_level",
    "run_full_benchmark",
    "eval_on_compilergym",
    "run_pipeline_on_compilergym",
    "evaluate_model_on_real_llvm",
    "print_results_table",
    "random_selector",
    "trm_selector",
    "RealLLVMEnv",
    "AVAILABLE_BENCHMARKS",
    "HAS_REAL_LLVM",
]
