# KV Cache Research Brief

Updated: 2026-03-31

This document is a research-oriented overview of KV cache for LLM inference. It is written as a standalone reference brief, not as a video script. The goal is to connect three layers that are often discussed separately:

- the transformer mechanics that make KV cache possible,
- the implementation details that make KV cache practical,
- the systems literature that turns KV cache into a first-class serving problem.

The emphasis is on public scientific papers and public implementation docs. Where a claim is an inference from multiple sources rather than a direct statement from one source, it is labeled as such.

## What KV Cache Is

In autoregressive transformer inference, each new output token is generated conditioned on all previous tokens. In a decoder-only transformer, the attention module for the current token needs access to the key and value tensors associated with earlier positions in the sequence. Those tensors are derived from the hidden states of earlier tokens and are expensive to recompute repeatedly.

The `KV cache` is the stored collection of those key (`K`) and value (`V`) tensors for prior tokens, layer by layer. Instead of recomputing the full attention state for the entire prefix at every decode step, the model computes the new query for the current position and attends over the already cached keys and values from earlier positions. That is the core reason autoregressive decoding is tractable at all for large models.

Without KV cache, each newly generated token would require recomputing key/value representations for the whole prefix across all decoder layers. This would make decoding dramatically more expensive and would repeatedly redo work that is already known from previous steps. With KV cache, the work for past tokens is reused; only the current token’s contributions need to be appended.

It is useful to distinguish two phases of inference:

- `Prefill`: the model processes the input prompt in parallel, computes hidden states for the prompt tokens, and builds the initial KV cache.
- `Decode`: the model generates one token at a time, reads the existing KV cache, computes the new token’s key/value tensors, appends them, and continues.

This distinction matters because KV cache is created mostly during prefill, but it becomes a dominant read path during decode.

The memory cost intuition is straightforward:

- cache size grows with `sequence length`,
- and with `number of layers`,
- and with `number of heads` or `number of KV heads`,
- and with `head dimension`,
- and with `precision` used to store K and V.

That is why long context can become memory-bound even when weights already fit comfortably into GPU memory.

## Transformer and Attention Foundation

The transformer foundation comes from *Attention Is All You Need* (Vaswani et al., 2017), which introduced scaled dot-product attention as the core primitive of the architecture. Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

A compact version of the attention equation is:

\[
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

For inference intuition, the most important part is not the equation itself but what the tensors mean:

- `Q` is the query of the current token or current positions being processed,
- `K` is the set of keys representing previously processed positions,
- `V` is the set of values carrying the information that attention will mix.

In decoder-only self-attention, keys and values for earlier tokens do not change once they have been computed for a given request. That is the structural reason caching works: after a token has been processed, its K and V can be reused in later decode steps.

Plain-English summary:

- each token produces a query, a key, and a value at every attention layer,
- the current token asks: “which earlier tokens matter to me?” using its query against cached keys,
- then it reads a weighted combination of cached values,
- and because earlier keys/values are stable for that request, they can be stored rather than recomputed.

This is also why KV cache is tightly tied to self-attention implementation, not to the model in the abstract.

## Implementation Intuition

The simplest way to think about KV cache implementation is:

- each request owns a growing set of cached K and V tensors,
- organized per layer,
- appended token by token during decode,
- later read repeatedly by future decode steps.

### Request-local KV cache

The baseline case is request-local caching: one request builds its own cache, uses it during decoding, and discards it when the request completes. Every LLM inference engine does some form of this by default.

### Sequence-major storage intuition

A naive implementation stores the cache for each request as a mostly contiguous sequence of blocks or tensors indexed by token position. This is intuitive, but it creates practical problems:

- requests have highly variable lengths,
- decode extends the cache incrementally,
- finished requests free memory in irregular patterns,
- long-context workloads produce large, uneven allocations.

This creates fragmentation pressure if the system assumes contiguous storage.

### Block/page-oriented storage intuition

Paged or block-oriented storage treats KV cache more like virtual memory than like one giant contiguous array. Instead of assuming that a request’s cache must be physically contiguous in GPU memory, the engine can store it in fixed-size blocks and map logical sequence positions onto non-contiguous physical memory.

That shift is one of the key insights behind the systems literature on LLM serving.

### Why contiguous allocation becomes a problem

Contiguous allocation is simple conceptually, but it becomes brittle when:

- many requests of different lengths coexist,
- some requests terminate early,
- some requests have long prompts and short outputs,
- others have short prompts and long outputs,
- prefix reuse needs to share cache blocks across requests.

If storage is too rigid, memory can be wasted even when the total free capacity would otherwise be enough.

### Why locality, fragmentation, reuse, and transfer matter

KV cache affects serving efficiency through several distinct mechanisms:

- `Locality`: decode repeatedly reads cached data; poor access patterns hurt performance.
- `Fragmentation`: irregular lifetimes and variable lengths waste memory if allocation is naive.
- `Reuse`: repeated prefixes or repeated prompt scaffolds make previously computed cache valuable.
- `Transfer`: if prefill and decode are split across workers or nodes, the cache itself becomes something that must be moved efficiently.

### Request-local cache vs reusable/shared prefix cache

These two ideas are related but distinct:

- `Request-local KV cache`: reuse within a single request across decode steps.
- `Reusable/shared prefix cache`: reuse between requests when they share the same prompt prefix.

The second is no longer just a transformer implementation detail; it becomes a systems and scheduling problem.

## Major Research Themes

## PagedAttention / memory management

Problem:
Naive KV-cache allocation wastes GPU memory because requests vary in length and lifetime.

Core idea:
Store KV cache in fixed-size blocks and map logical sequence positions to physical blocks, reducing fragmentation and improving utilization.

Why it mattered:
It turned KV cache from an implementation nuisance into a memory-management abstraction that could support practical large-scale serving.

What remained hard:
Paged cache management solves fragmentation and allocation efficiency, but it does not by itself solve prefix reuse policy, long-context transfer costs, or disaggregated serving.

## Prefix caching / cache reuse

Problem:
Many workloads repeatedly send prompts with large shared prefixes, such as system prompts, document context, or conversation history.

Core idea:
Reuse already computed KV cache for the shared prefix so prefill work can be skipped for subsequent requests.

Why it mattered:
It improves time-to-first-token and throughput for repeated-prefix workloads with very little effect on model outputs.

What remained hard:
Prefix reuse only helps when prefixes match; it introduces routing and eviction problems and does not reduce decode cost directly.

## Chunked prefill and scheduling

Problem:
Prefill and decode have very different resource profiles, and large prefills can interfere with ongoing decode traffic.

Core idea:
Split prompt processing into chunks and schedule it to avoid decode stalls or to better balance throughput and latency.

Why it mattered:
It reframed “prefill vs decode” as a scheduler problem, not just a batching problem.

What remained hard:
Chunk sizing, scheduling policy, and tail latency control remain workload-dependent and can interact poorly with other optimizations.

## Offload / hierarchical memory

Problem:
GPU HBM is limited, while KV cache can become enormous for long-context workloads.

Core idea:
Move KV cache, or colder portions of it, into lower tiers such as CPU memory or SSD-backed storage.

Why it mattered:
It extends feasible context length and effective cache capacity beyond GPU memory.

What remained hard:
The cost of bringing data back is real; offload is a capacity optimization, not a free latency optimization.

## Quantized KV cache

Problem:
KV cache can dominate memory consumption at long context lengths.

Core idea:
Store K and V in lower precision, reducing cache footprint and increasing the number of tokens that can fit in memory.

Why it mattered:
It directly targets one of the fastest-growing memory consumers in long-context inference.

What remained hard:
Errors can accumulate because cached values are reused over many future decode steps. KV cache is more sensitive than plain weight compression.

## Disaggregated serving / KV transfer

Problem:
Prefill and decode want different hardware and scheduling strategies. Co-locating them can create interference.

Core idea:
Separate prefill and decode onto different workers or instances and transfer the generated KV cache between them.

Why it mattered:
It allows prefill and decode to be optimized independently for TTFT and inter-token latency.

What remained hard:
KV transfer itself becomes a performance bottleneck, especially at long context; the system now depends on the quality of cache movement and placement.

## Key Papers

## Attention Is All You Need

- Year: 2017
- Link: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

Summary:
This paper introduced the transformer and the scaled dot-product attention mechanism that makes KV caching possible in modern autoregressive LLM inference. It did not frame the problem in serving terms, but it established the basic structure where each position produces queries, keys, and values.

Problem solved:
It replaced recurrent and convolutional sequence modeling with an attention-based architecture.

Why it mattered for KV cache:
KV cache is a direct operational consequence of the transformer’s attention structure.

What it did not solve:
It did not address inference-time memory growth, serving efficiency, paging, reuse, or distributed cache movement.

## Efficient Memory Management for Large Language Model Serving with PagedAttention

- Year: 2023
- Link: [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

Summary:
The PagedAttention paper introduced a memory-management abstraction for KV cache inspired by virtual memory paging. Instead of assuming contiguous allocation for each request’s KV cache, it stores KV cache in blocks that can be placed non-contiguously in physical GPU memory. The paper’s central contribution is not a new attention formula but a serving-oriented memory layout that dramatically improves utilization and throughput.

Problem solved:
It addressed waste from fragmentation and over-reservation in KV-cache-heavy serving systems.

Why it mattered:
This paper is the canonical turning point where KV cache became a first-class systems problem in LLM serving.

What it did not solve:
It did not by itself solve cross-request reuse policy, long-range disaggregation, or quantized cache storage.

## Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve

- Year: 2024
- Link: [https://arxiv.org/abs/2403.02310](https://arxiv.org/abs/2403.02310)

Summary:
Sarathi-Serve focuses on the throughput-latency tradeoff in LLM inference and introduces chunked prefills and scheduler design to reduce decode interference. The paper treats the prefill/decode asymmetry as a systems scheduling issue and argues that large prefill iterations can be partitioned and interleaved more intelligently with decode work.

Problem solved:
It addressed prefill/decode interference and poor utilization under mixed workloads.

Why it mattered:
It made chunked prefill a central concept in modern serving design and clarified how scheduling interacts with KV-cache creation.

What it did not solve:
It does not eliminate the underlying memory cost of KV cache; it manages when and how prefill work happens.

## Splitwise: Efficient Generative LLM Inference Using Phase Splitting

- Year: 2023 preprint / 2024 publication
- Link: [https://arxiv.org/abs/2311.18677](https://arxiv.org/abs/2311.18677)

Summary:
Splitwise characterizes prompt computation and token generation as two phases with distinct resource requirements and proposes splitting them across separate machines. The key systems insight is that prompt computation is compute-intensive while generation is more memory-intensive, so phase-specific hardware and deployment choices can improve efficiency.

Problem solved:
It tackled inefficiency caused by treating prefill and decode as if they wanted the same machine profile.

Why it mattered:
It helped establish phase splitting as a serious serving design, and KV cache is the state that must move across the split.

What it did not solve:
It does not by itself solve all the engineering challenges of large-scale cache transfer, prefix reuse, or disaggregated cache hierarchies.

## DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving

- Year: 2024
- Link: [https://arxiv.org/abs/2401.09670](https://arxiv.org/abs/2401.09670)

Summary:
DistServe argues that co-locating prefill and decode couples their resource allocation and creates strong interference. It proposes disaggregating the two and optimizing for goodput under service-level objectives. In this architecture, KV cache is not just local request state; it becomes an object that must be transferred efficiently from prefill to decode workers.

Problem solved:
It addressed poor latency/goodput outcomes caused by resource coupling between prefill and decode.

Why it mattered:
It made KV transfer a central systems problem and tied cache movement directly to quality-of-service goals.

What it did not solve:
It still depends on efficient interconnects, robust transfer mechanisms, and practical deployment support.

## Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads

- Year: 2024
- Link: [https://arxiv.org/abs/2401.11181](https://arxiv.org/abs/2401.11181)

Summary:
This paper, often discussed through the TetriInfer system, argues that mixed workloads suffer from interference when prefill and decode share resources too naively. It combines prompt partitioning, disaggregation, and scheduling with resource prediction to improve TTFT, job completion time, and efficiency per dollar.

Problem solved:
It addressed interference across mixed request types and heterogeneous workload conditions.

Why it mattered:
It broadened the serving discussion beyond single-objective throughput and showed that KV-cache-heavy serving design must respect workload diversity.

What it did not solve:
It does not remove the need for transfer-efficient cache design or universal reuse policy.

## Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

- Year: 2024
- Link: [https://arxiv.org/abs/2407.00079](https://arxiv.org/abs/2407.00079)

Summary:
Mooncake is explicitly KV-cache-centric. It separates prefill and decode clusters and uses underutilized CPU, DRAM, and SSD resources to build a disaggregated KV-cache hierarchy. The paper frames long-context serving as fundamentally constrained by cache capacity and movement, then designs scheduling and early rejection policies around that fact.

Problem solved:
It addressed the capacity and scheduling bottlenecks of long-context, cache-heavy production serving.

Why it mattered:
It is one of the clearest papers treating KV cache as the central resource around which serving architecture should be built.

What it did not solve:
Its ideas are especially compelling for large-scale industrial serving, but the architecture is more complex than the simpler public-engine deployment patterns common in open source.

## KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

- Year: 2024
- Link: [https://arxiv.org/abs/2402.02750](https://arxiv.org/abs/2402.02750)

Summary:
KIVI is a dedicated KV-cache quantization paper. Its core insight is that keys and values have different statistical properties and should not necessarily be quantized the same way. The paper proposes asymmetric 2-bit quantization with different strategies for keys and values, showing substantial cache compression while preserving model quality better than naive low-bit designs.

Problem solved:
It addressed the growing memory dominance of KV cache in long-context inference.

Why it mattered:
It made KV-cache quantization a standalone research topic rather than a footnote under general model quantization.

What it did not solve:
Its methods are more specialized than the simpler FP8-based approaches that are easier to integrate into general-purpose public runtimes today.

## KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization

- Year: 2024
- Link: [https://arxiv.org/abs/2401.18079](https://arxiv.org/abs/2401.18079)

Summary:
KVQuant studies ultra-long-context inference and argues that KV cache quickly becomes the dominant resource bottleneck. It explores sensitivity-aware and non-uniform KV-cache quantization schemes to extend feasible context length dramatically, targeting regimes where weight storage alone is no longer the main concern.

Problem solved:
It addressed the challenge of pushing context length far beyond what naive full-precision KV cache can support.

Why it mattered:
It made clear that long-context serving is often more a cache-capacity problem than a weight-capacity problem.

What it did not solve:
Like other aggressive KV-cache quantization work, it faces integration complexity and accuracy-risk tradeoffs relative to simpler production-friendly schemes.

## Additional Papers That Materially Improve Coverage

The following papers are not in the minimum list above, but they materially improve the systems picture of KV cache:

### Prompt Cache: Modular Attention Reuse for Low-Latency Inference

- Year: 2023
- Link: [https://arxiv.org/abs/2311.04934](https://arxiv.org/abs/2311.04934)

Why include it:
It extends the discussion from plain prefix reuse toward modular attention reuse, helping frame cache reuse as more than “entire-prefix match or nothing.”

### ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition

- Year: 2024
- Link: [https://arxiv.org/abs/2402.15220](https://arxiv.org/abs/2402.15220)

Why include it:
It directly connects prefix-aware KV cache ideas with chunked attention partitioning, making it relevant to both implementation and long-context scheduling discussions.

## Current Implementation Patterns

This section summarizes what is publicly visible in current tooling as of 2026-03-31. Where the claim is grounded in official docs, links are provided directly.

### Paper-important

These ideas are central to understanding the field, even if they are not all equally mainstream in day-to-day open-source deployment:

- Paged KV-cache management
- Prefix caching / cache reuse
- Chunked prefill
- Disaggregated prefill/decode
- KV-cache quantization
- Hierarchical cache storage across GPU and non-GPU memory

### Actually common in public tooling

These patterns are clearly present in public documentation and user-facing inference frameworks:

- `PagedAttention-style KV block management` in vLLM, originating from the PagedAttention paper
- `Automatic prefix caching` in vLLM
- `Quantized KV cache`, especially FP8 KV cache, in vLLM
- `Experimental disaggregated prefilling` in vLLM

Relevant official docs:

- vLLM Automatic Prefix Caching:
  - [https://docs.vllm.ai/en/latest/design/prefix_caching/](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- vLLM Quantized KV Cache:
  - [https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/)
- vLLM Disaggregated Prefilling:
  - [https://docs.vllm.ai/usage/disagg_prefill.html](https://docs.vllm.ai/usage/disagg_prefill.html)

Inference:
It is reasonable to say that public tooling has standardized around block/paged KV cache management and has meaningful support for prefix reuse and FP8 KV-cache quantization. It would be too strong to say that highly elaborate disaggregated cache hierarchies like Mooncake are “standard” in public open-source usage, even if the ideas are influential.

### Implementation details visible in public docs

From the vLLM prefix-caching docs:

- KV cache is handled in blocks, not just as one monolithic contiguous buffer.
- Reuse is implemented via hash-based block identity over tokens and prefix context.
- Prefix caching is explicitly described as skipping repeated prompt computation.

From the vLLM quantized-KV docs:

- FP8 KV cache is a user-facing feature.
- Both `e4m3` and `e5m2` options appear in public docs.
- Calibration of quantization scales is part of the documented workflow.

From the disaggregated-prefill docs:

- public vLLM guidance explicitly describes separate prefill and decode instances,
- and explicitly describes transferring KV cache between them,
- which makes the KV-transfer abstraction visible in public implementation, not just in papers.

## Open Problems and Tradeoffs

### TTFT vs TPS

Prefill-heavy optimizations and decode-heavy optimizations often pull in different directions. Improving time-to-first-token does not necessarily improve tokens-per-second, and vice versa.

### Cache reuse vs routing complexity

Prefix caching is powerful when workloads share repeated context, but it works best when related requests are routed to places where matching cache blocks are already hot. That introduces a systems-level placement and eviction problem.

### Long context vs memory growth

Longer context makes models more useful, but cache size scales with sequence length and can overwhelm HBM even when weights are already manageable.

### Quantization savings vs numerical error accumulation

KV-cache quantization can unlock substantial memory savings, but errors are reused over many future steps. This makes the quality-risk profile different from plain weight quantization.

### Disaggregation benefits vs KV transfer overhead

Separating prefill and decode can improve latency control and resource specialization, but the cache becomes a network-transfer object. At long context, moving the cache can become expensive enough to offset some of the gains.

## Takeaways

- KV cache exists because decoder self-attention would otherwise recompute the same historical key/value tensors at every decode step.
- Prefill creates the cache; decode repeatedly reads and extends it.
- In modern LLM serving, KV cache is often a bigger systems bottleneck than people expect from just looking at model weights.
- PagedAttention made KV cache a memory-management problem, not just an implementation detail.
- Prefix caching turns KV cache into a reuse problem across requests, not just within one request.
- Chunked prefill and disaggregated serving treat KV cache as part of a scheduling and transfer pipeline.
- KV-cache quantization matters because long-context inference can become dominated by cache footprint.
- Public tooling already reflects many of these ideas, but the most elaborate industrial cache architectures are still more influential than universally standard.

## Short Bibliography

- Vaswani et al., *Attention Is All You Need* (2017)  
  [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023)  
  [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

- Agrawal et al., *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (2024)  
  [https://arxiv.org/abs/2403.02310](https://arxiv.org/abs/2403.02310)

- Patel et al., *Splitwise: Efficient Generative LLM Inference Using Phase Splitting* (2023)  
  [https://arxiv.org/abs/2311.18677](https://arxiv.org/abs/2311.18677)

- Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving* (2024)  
  [https://arxiv.org/abs/2401.09670](https://arxiv.org/abs/2401.09670)

- Hu et al., *Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads* (2024)  
  [https://arxiv.org/abs/2401.11181](https://arxiv.org/abs/2401.11181)

- Qin et al., *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving* (2024)  
  [https://arxiv.org/abs/2407.00079](https://arxiv.org/abs/2407.00079)

- Liu et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache* (2024)  
  [https://arxiv.org/abs/2402.02750](https://arxiv.org/abs/2402.02750)

- Hooper et al., *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization* (2024)  
  [https://arxiv.org/abs/2401.18079](https://arxiv.org/abs/2401.18079)

- Gim et al., *Prompt Cache: Modular Attention Reuse for Low-Latency Inference* (2023)  
  [https://arxiv.org/abs/2311.04934](https://arxiv.org/abs/2311.04934)

- Ye et al., *ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition* (2024)  
  [https://arxiv.org/abs/2402.15220](https://arxiv.org/abs/2402.15220)
