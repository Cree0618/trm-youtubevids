# Local LLM Sources

Aktualizovano: 2026-03-11

Tento soubor je pomocny zdroj k `script.md`. Je urceny jako rychly prehled aktualnich faktu, benchmarku a odkazu k tematum:

- Apple Silicon a lokalni inference
- Mac Studio M3 Ultra 512 GB
- Mac mini versus Mac Studio
- local LLM stack pro Apple Silicon
- OpenClaw a agentic asistenti
- MiniMax M2.5
- Qwen 3.5
- Kimi K2.5

Poznamka k interpretaci:

- `Official` znamena vendor model card, vendor docs nebo Apple specs.
- `Community benchmark` znamena verejny benchmark nebo MLX quant card tretich stran. Je to uzitecne pro odhad realne rychlosti na Apple Silicon, ale neni to stejna vec jako vendor benchmark.
- Benchmarky mezi vendory nejsou dokonale apples-to-apples. Nejcistejsi primy srovnavaci zdroj je Qwen 3.5 card, protoze v tabulkach porovnava i Kimi K2.5. MiniMax M2.5 ma cast nejsilnejsich cisel ve vlastni evaluation sekci.

## 1. Hardware facts: Mac Studio M3 Ultra

### Official

- Apple u M3 Ultra uvadi az `512 GB` unified memory a `819 GB/s` memory bandwidth.
- Mac Studio s M3 Ultra ma `Thunderbolt 5` porty s rychlosti az `120 Gb/s`.
- V praxi to znamena, ze single-box lokalni inference na Apple Silicon je casto bottleneckovana spis memory bandwidth nez hrubym FLOPS marketingem.

Zdroje:

- https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/
- https://www.apple.com/mac-studio/specs/

### Mac mini versus Mac Studio

Official Apple specs jsou dulezite hlavne proto, ze internet casto smichava Mac mini a Mac Studio do jedne kategorie "Mac na AI".

Mac mini:

- M4: `120 GB/s`
- M4 Pro: `273 GB/s`
- az `64 GB` unified memory

Mac Studio s M3 Ultra:

- az `512 GB` unified memory
- `819 GB/s`
- Thunderbolt 5

Prakticky zaver:

- Mac mini a Mac Studio jsou dve ruzne tridy stroju
- to, co dava smysl na M3 Ultra 512 GB, casto vubec neni relevantni pro Mac mini

Zdroj:

- https://www.apple.com/mac-mini/specs/
- https://www.apple.com/mac-studio/specs/

## 2. MiniMax M2.5 na Apple Silicon

### Official

- MiniMax-M2.5 je open-weight model publikovany na Hugging Face.
- Official model card uvadi `229B` params.
- Official card zduraznuje silne software-engineering a agent benchmarky:
  - `SWE-bench Verified 80.2`
  - `Multi-SWE-bench 51.3`
  - `BrowseComp 76.3` s context management
  - `GPQA-Diamond 85.2`
  - `IFBench 70.0`
- MiniMax vlastni deployment guides jsou Linux-first a oficialne uvadeji hlavne SGLang, vLLM, Transformers a KTransformers.

Zdroje:

- https://huggingface.co/MiniMaxAI/MiniMax-M2.5
- https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/main/docs/transformers_deploy_guide.md

### Community benchmarky na M3 Ultra 512 GB

- `4-bit MLX`: zhruba `53 tok/s`
- `6.5-bit MLX`: zhruba `39 tok/s`
- `9-bit MLX`: zhruba `36.5 tok/s`

Prakticky z toho plyne:

- single-box MiniMax M2.5 na M3 Ultra 512 GB je realne zhruba `35-55 tok/s` podle quantizace
- to je velmi dobry pomer kvalita / rychlost na tak velky model
- na jednom stroji je to pouzitelne pro serious single-user local work

Zdroje:

- https://huggingface.co/MiniMaxAI/MiniMax-M2.5/discussions/7
- https://huggingface.co/inferencerlabs/MiniMax-M2.5-MLX-6.5bit
- https://huggingface.co/inferencerlabs/MiniMax-M2.5-MLX-9bit

## 3. 2 x Mac Studio M3 Ultra 512 GB s exo

### Official / vendor docs

- exo je open-source runtime pro distributed inference.
- exo README uvadi `up to 1.8x speedup on 2 devices`.
- MLX distributed docs popisuji `jaccl` backend a RDMA over Thunderbolt jako relevantni cestu pro tensor parallelism na Apple Silicon.

Prakticka interpretace:

- 2 boxy netvori jednu "kouzelnou" sdilenou RAM.
- Ziskavas dve oddelene 512GB pameti a inference musi byt shardovana pres link.
- Pro batch-1 decode je typicky limit spis synchronizacni latence a runtime overhead nez samotna hruba propustnost jednoho TB5 linku.

Rozumny odhad pro MiniMax M2.5 na `2 x M3 Ultra 512 GB` s exo a dobrym TB5/RDMA setupem:

- `4-bit`: cca `95-115 tok/s`
- `5-bit`: cca `75-90 tok/s`
- `6-6.5-bit`: cca `60-75 tok/s`
- `8-9-bit`: cca `45-57 tok/s`

To jsou decode odhady, ne prompt-ingest benchmarky.

Zdroje:

- https://github.com/exo-explore/exo
- https://ml-explore.github.io/mlx/build/html/usage/distributed.html
- https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5

## 4. Aktualni local LLM stack

Official nebo canonical zdroje, ktere dnes davaji smysl sledovat:

- `Ollama` pro jednoduchy local run a API server
- `llama.cpp` pro low-level runtime a sirsi kompatibilitu
- `LM Studio` pro GUI, local server a jednoduche testovani
- `MLX-LM` pro Apple Silicon heavy-duty use
- `Open WebUI` pro self-hosted UI, tools, RAG a multi-provider setup

Prakticky zaver:

- zacatecnik ma zacit v LM Studio nebo Ollama
- na Apple Silicon velkych modelu ma velkou hodnotu MLX ekosystem
- Open WebUI dava smysl jako vrstva nad vice runtime a provideru

Zdroje:

- https://ollama.com/
- https://github.com/ggml-org/llama.cpp
- https://lmstudio.ai/docs/app
- https://github.com/ml-explore/mlx-lm
- https://docs.openwebui.com/

## 5. Nejzajimavejsi modely na single M3 Ultra 512 GB

Nasledujici rychlosti jsou community MLX benchmarky nebo model-card claims publikovane pro M3 Ultra 512 GB. Ber je jako prakticke orientacni odhady, ne jako stejny test harness.

| Model | Typ | Prakticka rychlost | Poznamka |
| --- | --- | --- | --- |
| Qwen3.5-35B-A3B | multimodal MoE | `~96 tok/s` | rychlostni kral v Qwen 3.5 rodine |
| GLM-4.7-Flash | MoE | `~61-70 tok/s` | velmi silny rychly general-purpose model |
| Qwen3.5-122B-A10B | multimodal MoE | `~43.6 tok/s` | nejlepsi velky prakticky Qwen pro local |
| Qwen3.5-397B-A17B | multimodal MoE | `~40.2 tok/s` pri 4.1-bit | nejvetsi prakticky Qwen |
| MiniMax-M2.5 | text MoE | `~39 tok/s` pri 6.5-bit | velmi silny coding/search specialist |
| Kimi-K2.5 | multimodal MoE | `~26.8 tok/s` pri 3.6-bit | velmi tezky, ale silny agentic model |

Zdroje:

- https://huggingface.co/inferencerlabs/Qwen3.5-35B-A3B-MLX-5.5bit
- https://huggingface.co/inferencerlabs/GLM-4.7-Flash-MLX-5.5bit
- https://huggingface.co/inferencerlabs/GLM-4.7-Flash-MLX-6.5bit
- https://huggingface.co/inferencerlabs/Qwen3.5-122B-A10B-MLX-9bit
- https://huggingface.co/inferencerlabs/Qwen3.5-397B-A17B-MLX-4.1bit
- https://huggingface.co/inferencerlabs/MiniMax-M2.5-MLX-6.5bit
- https://huggingface.co/inferencerlabs/Kimi-K2.5-MLX-3.6bit

## 6. OpenClaw: co je aktualne dulezite

Official docs:

- OpenClaw je self-hosted AI gateway a personal agent system
- aktualni quickstart ukazuje doporuceny install script `curl -fsSL https://openclaw.ai/install.sh | bash`, `openclaw onboard --install-daemon`, `openclaw dashboard` a `openclaw gateway status`
- quickstart predpoklada `Node 22+`
- docs zduraznuji pairing, approvals, workstation a dashboard concepts
- local-model docs ukazuji workflow pres LM Studio a velky model, ne "maly model a hotovo"

Prakticky zaver:

- OpenClaw neni jen chat UI
- je to orchestrace, approvals, pairing, workspaces a routing
- local inference je jen jedna cast celeho systemu

Poznamka:

- OpenClaw docs aktualne ukazuji MiniMax M2.5 v local-model walkthrough. To neznamena, ze je to nutne jediny spravny model. Spis to ukazuje doporuceny workflow a to, ze pro agenty davaju smysl velke full-size modely.

Zdroje:

- https://docs.openclaw.ai/
- https://docs.openclaw.ai/quickstart
- https://docs.openclaw.ai/dashboard
- https://docs.openclaw.ai/concepts/pairing
- https://docs.openclaw.ai/concepts/approvals
- https://docs.openclaw.ai/concepts/workstation
- https://docs.openclaw.ai/gateway/local-models

## 7. Qwen 3.5 vs Kimi K2.5 vs MiniMax M2.5

### Qwen 3.5 family

Oficialni modely, ktere dava smysl srovnavat:

- `Qwen3.5-397B-A17B`
- `Qwen3.5-122B-A10B`
- `Qwen3.5-35B-A3B`
- `Qwen3.5-27B`

Co je dulezite:

- Qwen 3.5 je `native multimodal`.
- `397B-A17B` je nejsilnejsi all-round open Qwen.
- `122B-A10B` je nejlepsi kompromis velikost / kvalita / prakticnost.
- `27B` je kvalitnejsi compact model.
- `35B-A3B` je rychlejsi compact model.

Vybrane official benchmarky:

- `Qwen3.5-397B-A17B`
  - `GPQA 88.4`
  - `SWE-bench Verified 76.4`
  - `BFCL-V4 72.9`
  - `BrowseComp 69.0 / 78.6`
  - `MMMU-Pro 79.0`
- `Qwen3.5-122B-A10B`
  - `GPQA 86.6`
  - `SWE 72.0`
  - `BFCL 72.2`
  - `BrowseComp 63.8`
  - `MMMU-Pro 76.9`
- `Qwen3.5-27B`
  - `GPQA 85.5`
  - `SWE 72.4`
  - `MMMU-Pro 75.0`
- `Qwen3.5-35B-A3B`
  - `GPQA 84.2`
  - `SWE 69.2`
  - `BrowseComp 61.0`
  - `MMMU-Pro 75.1`

Zdroje:

- https://huggingface.co/Qwen/Qwen3.5-397B-A17B
- https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8
- https://huggingface.co/Qwen/Qwen3.5-27B
- https://huggingface.co/Qwen/Qwen3.5-35B-A3B

### Kimi K2.5

Co je dulezite:

- Kimi K2.5 je `native multimodal`.
- Official card ho prezentuje jako model pro long-horizon agent workflows, browser use, coding a vizualni ulohy.
- Je to ale velmi tezky model pro local run, hlavne kvuli aktivnim parametrum a footprintu.

Vybrane official benchmarky:

- `HLE w/tools 50.2`
- `SWE-bench Verified 76.8`
- `BrowseComp 74.9` s context management
- `BrowseComp 78.4` s Agent Swarm
- `MMMU-Pro 78.5`
- `GPQA 87.6`

Zdroj:

- https://huggingface.co/moonshotai/Kimi-K2.5

### MiniMax M2.5

Co je dulezite:

- open release je textovy model, ne multimodalni VLM
- official card ukazuje velmi silne software-engineering a search benchmarky
- na standardnich reasoning benchmarcich nevypada tak silne jako flagship Qwen 3.5 nebo Kimi K2.5

Vybrane official benchmarky:

- `SWE-bench Verified 80.2`
- `Multi-SWE-bench 51.3`
- `BrowseComp 76.3` s context management
- `GPQA-Diamond 85.2`
- `AIME 2025 86.3`
- `HLE 19.4`

Zdroj:

- https://huggingface.co/MiniMaxAI/MiniMax-M2.5

### Prakticky zaver

- Pokud chces `nejlepsi open all-round multimodal model`, nejzajimavejsi je `Qwen3.5-397B-A17B`.
- Pokud chces `nejlepsi prakticky large model na lokalni stroj`, nejzajimavejsi je `Qwen3.5-122B-A10B`.
- Pokud chces `nejrychlejsi solidni Qwen`, dava smysl `Qwen3.5-35B-A3B`.
- Pokud chces `silny coding/search specialist`, MiniMax M2.5 vypada velmi dobre.
- Pokud chces `agentic multimodal monster` a nevadi ti obrovska narocnost, Kimi K2.5 je velmi silny, ale nejhur se provozuje lokalne.

## 8. Hlavni bottlenecks pro local inference

- `memory bandwidth` je casto dulezitejsi nez marketing FLOPS
- `aktivni parametry` jsou pro decode casto dulezitejsi nez total params
- `KV cache` roste s contextem a zere dalsi unified memory
- `runtime maturity` na Apple Silicon dela realny rozdil
- `batch throughput` a `single-user latence` nejsou stejna vec
- model se muze "vejit", ale porad byt neprijemne pomaly

## 9. Practical talking points pro script

- "Utahne to model" neni stejna vec jako "pobezí to rozumne rychle".
- U lokalni AI je potreba rikat `model`, `quant`, `context`, `tok/s` a `use case`.
- Mac Mini / Mac Studio hype ma realny zaklad v unified memory a bandwidth, ale casto se prehani.
- Nejvetsi rozdil casto neni mezi "bezi / nebezi", ale mezi "technicky se spusti" a "je to pouzitelne".
