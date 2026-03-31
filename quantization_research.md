# Quantization Research Dossier

Aktualizovano: 2026-03-28

Tento dokument mapuje relevantni landscape kvantizace pro LLM inference a navazujici fine-tuning. Je psany pro technicky gramotne publikum a rozlisuje mezi:

- `paper-important`: dulezite metodicke body ve vyvoji oboru
- `actually used`: metody a formaty, ktere se realne objevuji v toolingu a deploymentu v letech 2024-2026

Hlavni zamer:

- inference-first pohled
- proc kvantizace funguje jako systemovy nastroj
- proc ruzne casti modelu snaseji nizkou presnost ruzne
- jak se meni obrazek u KV cache
- kde se v praxi pouziva AWQ / GPTQ / bitsandbytes / GGUF / FP8

## 1. Core Framing

### Co je kvantizace

Kvantizace je zpusob, jak reprezentovat vahy, aktivace nebo cache v mensim poctu bitu. Misto BF16 nebo FP16 se pouzije INT8, FP8, INT4, FP4 nebo jina nizkopresna reprezentace. Cilem neni "udelat model mensi" v abstraktnim smyslu, ale zmenit pomer mezi:

- kapacitou pameti
- propustnosti pameti
- dostupnym compute
- presnosti numeriky

V inference neni hlavni otazka jen "kolik parametru ma model", ale:

- kolik bytu je potreba nacist pri decode
- kolik VRAM zabiraji vahy
- kolik VRAM zabira KV cache
- ktere operace jsou compute-bound a ktere memory-bound

### Proc kvantizace pomaha

Pro LLM inference plati uzitecne pravidlo:

- `prefill` byva casto vic compute-bound
- `decode` byva casto vic memory-bound

Kdyz snizis precision:

- prefill muze tez profitovat z rychlejsich low-precision kernelu
- decode casto profituje hlavne z mensiho objemu dat, ktere je treba nacitat z pameti

To je duvod, proc je kvantizace v inference tak silna paka i bez toho, aby menila samotnou architekturu modelu.

### Proc kvantizace neni jedna technika

Pod slovem "quantized model" se v praxi schovava nekolik ruznych veci:

- weight-only PTQ
- weight+activation PTQ
- KV-cache quantization
- kvantizace pri fine-tuningu
- nativni low-precision floating-point formaty pro inference engine a hardware

Dve 4-bit varianty stejneho modelu mohou mit velmi odlisnou kvalitu i rychlost, protoze se lisi:

- metoda hledani scale / zero-point / codebooku
- granularita: per-tensor, per-channel, per-group, blockwise
- to, zda zustavaji citlive kanaly nebo outliery ve vyssi presnosti
- inference kernel a format checkpointu

## 2. Taxonomy

### Weight-only PTQ

Kvantizuje pouze vahy, aktivace zustavaji typicky ve vyssi presnosti. Prakticky velmi dulezite pro local inference, protoze:

- dramaticky snizi memory footprint
- casto je jednodussi na deployment
- mnohdy zachova kvalitu lepe nez agresivni W4A4 / W4A8

Typicke metody:

- LLM.int8()
- GPTQ
- AWQ
- HQQ
- AQLM
- QuIP#
- AutoRound / SignRound

### Weight+Activation PTQ

Kvantizuje vahy i aktivace, casto s cilem zrychlit i compute-bound casti inference a ne jen snizit memory footprint. Historicky sem patri hlavne:

- SmoothQuant
- QQQ
- moderni FP8 pipelines
- end-to-end 4-bit linie typu QuaRot a SpinQuant

### KV-cache quantization

Specialni podkategorie. KV cache neni staticka jako vahy:

- roste s delkou kontextu
- je ctana opakovane pri decode
- chyba se muze kumulovat pres dalsi tokeny

Proto ma KV cache vlastni paper lineage:

- KIVI
- KVQuant

V praxi se dnes casto objevuje hlavne:

- FP8 KV cache ve vLLM

### Quantization-aware training

Model je trenovan nebo dotrenovan s vedomim, ze pobezi v nizke presnosti. To sem patri spise konceptualne; pro open-model deployera je praktictejsi PTQ nebo low-bit fine-tuning. Dulezite je vedet:

- QAT casto vede k lepsim ultra-low-bit vysledkum
- ale je mnohem drazsi a mene univerzalni nez PTQ

### Low-bit fine-tuning

Sem patri hlavne QLoRA. Neni to primarne inference paper, ale je zasadni pro pochopeni, proc 4-bit reprezentace zdomacnely:

- model muze byt frozen a 4-bit
- adaptery se dotrenuji nad nim
- dramaticky klesa memory budget pro fine-tuning

### Native low-precision formats / hardware context

V deploymentu uz nejde jen o "algoritmus kvantizace", ale i o to, co umi hardware a runtime:

- INT8
- FP8
- MXFP8
- FP4 / MXFP4 / NVFP4

To je dulezite hlavne pro server-side inference, kde low-precision format muze byt primo podporovan tensor cores a inference enginy.

## 3. Mechanisms You Need To Explain In The Video

### Scale a zero-point

Nejjednodussi predstava kvantizace:

1. puvodni floating-point hodnoty se mapuji do mensi mnoziny reprezentovatelnych hodnot
2. k tomu je potreba `scale`
3. u nekterych integer schemat i `zero-point`

Prakticky:

- mensi scale = jemnejsi rozliseni v male oblasti
- vetsi range = horsi jemnost
- outliery kazi efektivitu jednoduche kvantizace

### Granularita

Stejna metoda muze fungovat velmi jinak podle toho, jak hrube kvantizujes:

- `per-tensor`: jedna skala pro cely tensor
- `per-channel`: jina skala pro kanal
- `per-group`: jina skala pro skupinu vah
- `blockwise`: jina skala pro maly blok

Vyssi granularita:

- casto zlepsi kvalitu
- ale pridava metadata a casto i runtime overhead

### Integer vs floating low-bit formats

`INT4/INT8`:

- casto velmi kompaktni
- vyzaduji scale a casto zero-point
- mohou byt velmi efektivni, kdyz na ne existuji dobre kernely

`FP8/FP4`:

- lepe pracuji s dynamickym rozsahem
- casto vhodnejsi pro aktivace a nektere runtime pipelines
- jsou silne vazane na podporu hardwaru

### Codebooks a vector quantization

Ne vsechny metody delaji prostou scalar quantization. AQLM nebo QuIP# kvantizuji skupiny vah spis jako vektorovy problem:

- vice vah najednou
- kodovani pres codebook
- lepsi komprese v extrémne nizkych bitech
- casto slozitejsi runtime a horsi univerzalni adopce

### Proc nektere casti modelu trpi vic

Od nejmensiho po nejvetsi riziko:

1. linearni vahy
2. cast aktivaci
3. KV cache
4. attention internals a softmax-adjacent numerika

Intuice:

- vahy jsou staticke a lze je offline analyzovat
- aktivace zavisi na datech a casto maji outliery
- KV cache se pouziva opakovane
- attention je citliva na male numericke odchylky, ktere ovlivnuji vahovani kontextu

## 4. Main Paper Families

## 4.1 LLM.int8()

- Exact title: `LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale`
- Year: 2022
- Canonical link: https://arxiv.org/abs/2208.07339
- Secondary link: https://huggingface.co/papers/2208.07339
- Thesis: Int8 inference lze delat bez ztraty kvality, pokud oddelis emergent outliery a zpracujes je ve vyssi presnosti.
- What is quantized: predevsim linear layers v 8-bit pipeline s mixed-precision zpracovanim outlier dimensions.
- Calibration/training: bez retrainu; metodicky jde o specializovanou inference proceduru.
- Bit-widths / formats: int8 + fp16 fallback pro outliery.
- Main tradeoff: velmi dobra dostupnost a robustni kvalita, ale mensi kompresni pomer nez 4-bit rodiny.
- Why it mattered: zpopularizovalo myslenku, ze hlavni problem nejsou jen bity, ale outliery v transformerech.
- What replaced or limited it: pro maximalni memory savings ho prekonaly 4-bit weight-only metody a QLoRA pro fine-tuning.
- Practical relevance now: stale velmi relevantni jako mental model a v bitsandbytes ecosystemu.

## 4.2 GPTQ

- Exact title: `GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers`
- Year: 2022
- Canonical link: https://arxiv.org/abs/2210.17323
- Secondary link: https://huggingface.co/papers/2210.17323
- Thesis: one-shot PTQ s aproximaci druheho radu dovoluje presne weight-only 3-4bit kvantizace bez retrainingu.
- What is quantized: vahy, typicky weight-only.
- Calibration/training: potrebuje kalibracni data / calibration pass.
- Bit-widths / formats: typicky 4-bit, take 3-bit a nizsi rezimy.
- Main tradeoff: velmi silna kvalita pro W4A16, ale inference kvalita a rychlost zavisi na konkretnim kernel stacku.
- Why it mattered: stalo se jednim z dominantnich PTQ pristupu v open-source LLM deploymentu.
- What replaced or limited it: AWQ byva casto preferovano pro nektere deployment workflows; AutoRound a dalsi moderni PTQ pristupy tlaci kvalitu dal.
- Practical relevance now: `actually used`.

## 4.3 SmoothQuant

- Exact title: `SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models`
- Year: 2022
- Canonical link: https://arxiv.org/abs/2211.10438
- Secondary link: https://huggingface.co/papers/2211.10438
- Thesis: presune obtiznost kvantizace z aktivaci do vah pomoci algebraicky ekvivalentni transformace a umozni W8A8.
- What is quantized: vahy i aktivace.
- Calibration/training: post-training, typicky s kalibracnim datasetem.
- Bit-widths / formats: hlavne INT8 W8A8.
- Main tradeoff: vyborne pro hardware-friendly W8A8, ale neni to standardni cesta pro ultra-kompaktni local 4-bit deployment.
- Why it mattered: definovalo hlavni linii activation-aware W8A8 PTQ pro LLM inference.
- What replaced or limited it: pro local memory savings casto nestaci 8-bit; pro 4-bit end-to-end je potreba jina trida metod.
- Practical relevance now: `paper-important` a stale relevantni v FP8/W8A8 mental modelu.

## 4.4 AWQ

- Exact title: `AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration`
- Year: 2023
- Canonical link: https://arxiv.org/abs/2306.00978
- Secondary link: https://huggingface.co/papers/2306.00978
- Thesis: chrani malou sadu saliencnich vah podle aktivacni statistiky, takze 4-bit weight-only quantization drzi kvalitu a je hardware-friendly.
- What is quantized: vahy, typicky W4A16 weight-only.
- Calibration/training: potrebuje calibration, ale ne retraining.
- Bit-widths / formats: typicky 4-bit.
- Main tradeoff: velmi dobry pomer kvalita / prakticka inference / hub adoption.
- Why it mattered: stalo se jednim z nejpouzivanejsich formatovych a deployment standardu pro open LLM checkpoints.
- What replaced or limited it: neni to end-to-end low-bit pipeline pro aktivace a KV cache; neni to final answer pro ultra-low-bit rezimy.
- Practical relevance now: `actually used`, zejmena v HF a nekterych GPU deployment pipelines.

## 4.5 QLoRA

- Exact title: `QLoRA: Efficient Finetuning of Quantized LLMs`
- Year: 2023
- Canonical link: https://arxiv.org/abs/2305.14314
- Secondary link: https://huggingface.co/papers/2305.14314
- Thesis: frozen 4-bit model + LoRA adaptery = dramaticky levnejsi fine-tuning bez velke ztraty kvality.
- What is quantized: frozen base model pri treningu / fine-tuningu.
- Calibration/training: training-adjacent metoda, ne cista inference PTQ.
- Bit-widths / formats: NF4, double quantization.
- Main tradeoff: neni to primarni inference paper, ale zmenilo to adopci 4-bit ekosystemu.
- Why it mattered: 4-bit prestal byt jen inference trik a stal se zakladnim stavebnim kamenem PEFT workflow.
- What replaced or limited it: pro cistou inference je casto dulezitejsi AWQ/GPTQ/GGUF/FP8 pipeline.
- Practical relevance now: `actually used` pro PEFT a adaptation workflows.

## 4.6 HQQ

- Exact title: `Half-Quadratic Quantization` (v dostupnych zdrojich jsem overil oficialni implementaci a dokumentaci, ne primarni arXiv preprint)
- Year: practical release line 2023-2025
- Canonical links:
  - https://github.com/dropbox/hqq
  - https://huggingface.co/docs/transformers/quantization/hqq
  - https://dropbox.tech/machine-learning/halfquadratic-quantization-of-large-machine-learning-models
- Thesis: rychla calibration-free quantization pres robustni optimalizaci, vhodna i pro velmi nizke bity.
- What is quantized: typicky vahy; v praxi casto on-the-fly replacement linear layers.
- Calibration/training: calibration-free.
- Bit-widths / formats: 8, 4, 3, 2, 1 bit.
- Main tradeoff: velmi atraktivni flexibilita a rychlost kvantizace, ale mensi standardizace nez GPTQ/AWQ.
- Why it mattered: ukazalo, ze kalibrace neni nutna podminka pro pouzitelnou PTQ kvalitu.
- What replaced or limited it: mensi ekosystem checkpointu a mensi "default status" v mainstream deploymentu.
- Practical relevance now: `actually used`, ale spise jako flexibilni specializovana volba nez universalni default.

## 4.7 AQLM

- Exact title: `Extreme Compression of Large Language Models via Additive Quantization`
- Year: 2024
- Canonical link: https://arxiv.org/abs/2401.06118
- Secondary links:
  - https://huggingface.co/papers/2401.06118
  - https://huggingface.co/docs/transformers/en/quantization/aqlm
- Thesis: multi-codebook additive quantization tlaci kvalitu v ekstremne nizkych bit-ratech, zejmena kolem 2 bitu.
- What is quantized: vahy, vektorove / codebookove.
- Calibration/training: PTQ s vlastni optimalizaci.
- Bit-widths / formats: ultra-low-bit, typicky 2-3 bit ekvivalenty.
- Main tradeoff: skvela komprese, ale slozitejsi runtime a ne univerzalni nejrychlejsi deployment path.
- Why it mattered: posunulo hranici toho, co jde delat pod 3 bity, bez uplneho kolapsu kvality.
- What replaced or limited it: pro siroky deployment byva jednodussi zustat u GPTQ/AWQ/GGUF nebo FP8.
- Practical relevance now: `paper-important`, castecne `actually used` pres HF integraci, ale ne mainstream default.

## 4.8 QuIP#

- Exact title: `QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks`
- Year: 2024
- Canonical link: https://arxiv.org/abs/2402.04396
- Secondary link: https://huggingface.co/papers/2402.04396
- Thesis: randomizovane Hadamard rotations + lattice codebooks + fine-tuning pro silne ultra-low-bit weight-only PTQ.
- What is quantized: vahy.
- Calibration/training: PTQ plus jemne doladeni fidelity.
- Bit-widths / formats: do 4 bit a niz.
- Main tradeoff: velmi silne vysledky v extremni kompresi, ale vyssi komplexita a mensi jednoducha adopce.
- Why it mattered: ukazalo, ze rotations + vector quantization jsou velmi silna kombinace.
- What replaced or limited it: slozitejsi pipeline a mensi deployment jednoduchost.
- Practical relevance now: `paper-important`.

## 4.9 AutoRound / SignRound

- Exact title: `Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs`
- Year: 2023
- Canonical link: https://arxiv.org/abs/2309.05516
- Secondary links:
  - https://huggingface.co/papers/2309.05516
  - https://huggingface.co/blog/autoround
- Thesis: rounding decisions jsou casto dulezitejsi nez slozite perturbace; signed gradient descent umi rychle optimalizovat rounding pri weight-only PTQ.
- What is quantized: vahy.
- Calibration/training: lehka tuning-based PTQ.
- Bit-widths / formats: typicky 4-bit a 2-bit rezimy.
- Main tradeoff: dobre practical accuracy improvements bez inference overheadu; ekosystem dnes stoji i na serializaci do AWQ/GPTQ-like formatu.
- Why it mattered: premostilo research rounding ideje k industrialnejsimu toolingu.
- What replaced or limited it: stale je to hlavne specialized offline quantization workflow, ne univerzalni runtime standard.
- Practical relevance now: roste, zejmena v Intel / HF / cross-format workflows; `emerging actually used`.

## 4.10 QuaRot

- Exact title: `QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs`
- Year: 2024
- Canonical link: https://arxiv.org/abs/2404.00456
- Secondary link: https://huggingface.co/papers/2404.00456
- Thesis: vhodne rotace odstrani outliery bez zmeny fp chovani a umozni end-to-end 4-bit quantization vcetne vah, aktivaci a KV cache.
- What is quantized: vahy, aktivace i KV cache.
- Calibration/training: PTQ s rotational transform.
- Bit-widths / formats: end-to-end 4-bit.
- Main tradeoff: metodicky krasne reseni outlier problemu, ale deployment pipeline je komplexnejsi nez weight-only standardy.
- Why it mattered: posunulo conversation od "weights only" k realnemu W4A4KV4-like thinking.
- What replaced or limited it: zatim omezeny mainstream deployment v porovnani s AWQ/GPTQ/FP8.
- Practical relevance now: `paper-important`.

## 4.11 SpinQuant

- Exact title: `SpinQuant: LLM quantization with learned rotations`
- Year: 2024
- Canonical link: https://arxiv.org/abs/2405.16406
- Secondary link: https://huggingface.co/papers/2405.16406
- Thesis: ne vsechny rotace jsou stejne dobre; naucene rotace zlepsuji end-to-end PTQ pro weights, activations i KV cache.
- What is quantized: vahy, aktivace, KV cache.
- Calibration/training: PTQ s learned rotations nad validacni sadou.
- Bit-widths / formats: 4-bit regime.
- Main tradeoff: jeste lepsi outlier handling nez fixed rotation pristupy, ale vyssi metodicka slozitost.
- Why it mattered: ukazalo, ze rotation space itself je optimalizacni problem.
- What replaced or limited it: deployment complexity a mensi mainstream tooling adoption.
- Practical relevance now: `paper-important`.

## 4.12 KIVI

- Exact title: `KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache`
- Year: 2024
- Canonical link: https://arxiv.org/abs/2402.02750
- Secondary links:
  - https://huggingface.co/papers/2402.02750
  - https://arxiv.gg/abs/2402.02750
- Thesis: klice a hodnoty maji odlisnou distribuci, proto se maji kvantizovat odlisne: key per-channel, value per-token.
- What is quantized: KV cache.
- Calibration/training: tuning-free.
- Bit-widths / formats: asymetricka 2-bit KV-cache quantization.
- Main tradeoff: velmi silne memory savings a throughput gains, ale uzka specializace na KV cache.
- Why it mattered: dalo KV cache quantization samostatnou a presvedcivou teoretickou i praktickou identitu.
- What replaced or limited it: mainstream runtimy zatim casteji nasazuji jednodussi FP8 KV cache nez 2-bit specializovane pipeline.
- Practical relevance now: `paper-important`, specializovane deployment use-cases.

## 4.13 KVQuant

- Exact title: `KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization`
- Year: 2024
- Canonical link: https://arxiv.org/abs/2401.18079
- Secondary link: https://slice.eecs.berkeley.edu/papers/kvquant-towards-10-million-context-length-llm-inference-with-kv-cache-quantization/
- Thesis: ultra-long-context inference je limitovana KV cache, a proto je treba specializovana KV-cache quantization se sensitivity-aware a non-uniform postupy.
- What is quantized: KV cache.
- Calibration/training: post-training specializovana KV quantization.
- Bit-widths / formats: agresivni nizkobitove KV-cache rezimy vcetne 3-bit.
- Main tradeoff: skvela demonstrace pro long context, ale deployment complexity je vyssi nez jednoduche FP8 store/dequantize pipeline.
- Why it mattered: ukazalo, ze KV cache muze byt dominantni bottleneck driv nez samotne vahy.
- What replaced or limited it: bezny production stack dnes casto preferuje jednodussi FP8 KV cache a paging.
- Practical relevance now: `paper-important`.

## 4.14 QQQ

- Exact title: `QQQ: Quality Quattuor-Bit Quantization for Large Language Models`
- Year: 2024
- Canonical link: https://arxiv.org/abs/2406.09904
- Secondary link: https://arxiv.org/pdf/2406.09904
- Thesis: W4A8 muze byt prakticka cesta k akceleraci obou fazi inference, pokud se zlepsi smoothing a kompenzace chyby.
- What is quantized: vahy a aktivace.
- Calibration/training: PTQ s kompenzaci a kernel engineeringem.
- Bit-widths / formats: 4-bit weights, 8-bit activations.
- Main tradeoff: ambiciozni sweet spot mezi W8A8 a W4A16, ale silne zavisly na kernel stacku.
- Why it mattered: premostuje quality-vs-speed mezeru mezi kompresi a realnou akceleraci.
- What replaced or limited it: neni zatim default deployment standardem.
- Practical relevance now: `paper-important`, bridge method.

## 5. Mention But Do Not Center

Tyto metody a smery se vyplati zminit, ale neni nutne je delat osou 10-12min videa:

- `QUIK`: hybridni 4-bit weight+activation pipeline s outlier retention
- `SpQR`: sparse-quantized representation
- `BitNet / 1-bit native models`: dulezite tema, ale je to jina kategorie nez post-training quantization otevrenych checkpointu
- `EETQ`, `Quanto`, `torchao`, `compressed-tensors`, `FBGEMM`, `Quark`: dulezite jako tooling a kernel / serialization vrstva, ne jako hlavni paper narrative

## 6. What Is Actually Used In Practice

Toto je synteticky zaver z official docs a ecosystemu k datu 2026-03-28.

### Local inference

Nejcasteji narazis na:

- `GGUF / llama.cpp`
- `bitsandbytes` 8-bit a 4-bit
- `GPTQ`
- `AWQ`

Prakticka poznamka:

- local users casto neresi "nejlepsi paper", ale checkpoint format + kompatibilitu s runtime
- GGUF kvantizace je v praxi obrovsky dulezita, i kdyz paper narrative kolem ni neni tak centralni jako kolem GPTQ/AWQ

Sources:

- llama.cpp tensor encodings:
  - https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes
- Transformers quantization overview:
  - https://huggingface.co/docs/transformers/en/main_classes/quantization
- bitsandbytes docs:
  - https://huggingface.co/docs/transformers/v4.51.3/quantization/bitsandbytes

### Server inference

Na serveru dnes dava smysl delit deployment do dvou hlavnich svetov:

- `weight-only low-bit checkpoints` jako AWQ / GPTQ
- `hardware-native low precision` jako FP8 a navazujici KV-cache quantization

Prakticky dnes pusobi nejsilneji:

- `FP8` pro vahy/aktivace tam, kde je vhodny hardware a engine
- `FP8 KV cache` ve vLLM
- `AWQ/GPTQ` pro checkpoint-centric deployment

Sources:

- vLLM supported quantization matrix:
  - https://docs.vllm.ai/en/v0.10.1.1/features/quantization/supported_hardware.html
- vLLM quantized KV cache:
  - https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html
  - https://docs.vllm.ai/usage/quantization/quantized_kvcache/
- NVIDIA Transformer Engine overview:
  - https://docs.nvidia.com/deeplearning/transformer-engine/

### Kde je hranice mezi paper-important a actually used

`Actually used today`:

- LLM.int8 / bitsandbytes
- GPTQ
- AWQ
- QLoRA
- GGUF-style quantized checkpoints
- FP8 inference
- FP8 KV cache

`Important academically, but not default deployment choices`:

- SmoothQuant
- QuaRot
- SpinQuant
- QuIP#
- AQLM
- KIVI
- KVQuant
- QQQ

`Hybrid / emerging`:

- HQQ
- AutoRound

## 7. Hardware-Native Precision Context

Pro video je dulezite vysvetlit rozdil mezi:

- algoritmickou PTQ metodou
- storage/checkpoint formatem
- a precision formatem, ktery umi hardware akcelerovat

### FP8

FP8 je dnes relevantni deployment precision pro inference i training na modernich NVIDIA generacich. Neni to "jen dalsi quantized checkpoint", ale nativni low-precision floating-point cesta.

Sources:

- Transformer Engine:
  - https://docs.nvidia.com/deeplearning/transformer-engine/

### MXFP8 a NVFP4

Tohle je spise deployment context nez hlavni paper rodina pro video. Dulezite body:

- jemnejsi block-level scaling
- lepsi accuracy / efficiency tradeoff na Blackwell
- relevance hlavne pro inference engineering, ne pro casual local users

Sources:

- NVFP4 overview:
  - https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/

Inference:

Pouziti MXFP8 / NVFP4 je silne zavisle na hardware a runtime stacku. Pro video je lepsi je zminit jako "kam smeruje industrial inference" nez jako defaultni radu pro bezneho open-source deployera.

## 8. What The Script Must Emphasize

### Point 1

Kvantizace neni jen komprese. Je to zmena numericke reprezentace, ktera meni:

- model size
- pametove presuny
- fit do VRAM
- throughput
- nekdy i kernel path

### Point 2

Ne vsechno v modelu je stejne citlive:

- vahy jsou nejsnazsi vstup
- aktivace jsou horsi
- KV cache je samostatny problem
- attention je rizikova oblast

### Point 3

Nejvetsi prakticka chyba je mluvit o "4-bit quantized model" jako o jedne veci. Je potreba se ptat:

- jaka metoda?
- co presne je quantized?
- jaka granularita?
- jaky runtime?
- jaka kvalita po evaluaci?

### Point 4

KV cache si zaslouzi vlastni cast. U dlouheho kontextu se casto zmeni z "detailu attention implementace" na hlavni memory bottleneck inference.

## 9. Recommended Video Narrative

Pro 10-12 minutovy script je nejlepsi tato osa:

1. Quantization jako systems problem
2. Prosim zadna magie: scale, group, outlier, precision
3. Co kvantizujeme: weights, activations, KV cache
4. Historicka kostra:
   - LLM.int8
   - GPTQ / AWQ
   - SmoothQuant
   - QLoRA
   - moderni PTQ frontier: HQQ, AQLM, QuIP#, AutoRound, QuaRot, SpinQuant
5. KV cache jako samostatny bottleneck:
   - KIVI
   - KVQuant
   - proc dnes v praxi casto vyhrava jednodussi FP8 KV cache
6. Co se skutecne nasazuje:
   - AWQ
   - GPTQ
   - bitsandbytes
   - GGUF
   - FP8
7. Ktere marketingove zkratky jsou podezrele

## 10. Canonical Source List

### Papers

- LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
  - https://arxiv.org/abs/2208.07339
- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
  - https://arxiv.org/abs/2210.17323
- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
  - https://arxiv.org/abs/2211.10438
- QLoRA: Efficient Finetuning of Quantized LLMs
  - https://arxiv.org/abs/2305.14314
- AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
  - https://arxiv.org/abs/2306.00978
- Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs
  - https://arxiv.org/abs/2309.05516
- Extreme Compression of Large Language Models via Additive Quantization
  - https://arxiv.org/abs/2401.06118
- KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization
  - https://arxiv.org/abs/2401.18079
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
  - https://arxiv.org/abs/2402.02750
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks
  - https://arxiv.org/abs/2402.04396
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
  - https://arxiv.org/abs/2404.00456
- SpinQuant: LLM quantization with learned rotations
  - https://arxiv.org/abs/2405.16406
- QQQ: Quality Quattuor-Bit Quantization for Large Language Models
  - https://arxiv.org/abs/2406.09904

### Tooling and runtime docs

- Transformers quantization overview
  - https://huggingface.co/docs/transformers/en/main_classes/quantization
- bitsandbytes docs
  - https://huggingface.co/docs/transformers/v4.51.3/quantization/bitsandbytes
- HQQ docs
  - https://huggingface.co/docs/transformers/quantization/hqq
- AQLM docs
  - https://huggingface.co/docs/transformers/en/quantization/aqlm
- llama.cpp tensor encodings
  - https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes
- vLLM supported hardware / quantization
  - https://docs.vllm.ai/en/v0.10.1.1/features/quantization/supported_hardware.html
- vLLM quantized KV cache
  - https://docs.vllm.ai/usage/quantization/quantized_kvcache/

### Vendor / hardware context

- NVIDIA Transformer Engine
  - https://docs.nvidia.com/deeplearning/transformer-engine/
- NVIDIA NVFP4 blog
  - https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/

## 11. Bottom-Line Research Conclusion

Pokud ma video dat dospelemu technickemu publiku opravdovou hodnotu, mela by byt hlavni teze tato:

`Quantization neni jeden trik, ale cela rodina kompromisu mezi kvalitou, pameti, bandwidthem, kernel supportem a deployment ergonomii.`

Historicky nejdulezitejsi body:

- LLM.int8() vysvetlilo outliery
- GPTQ a AWQ ovladly weight-only PTQ praxi
- SmoothQuant definovalo activation-aware W8A8 smer
- QLoRA zpopularizovalo 4-bit i mimo inference
- QuaRot a SpinQuant ukazaly, ze outlier problem lze resit rotationalne
- KIVI a KVQuant oddelily KV cache jako vlastni quantization problem
- dnesni production praxe se opira hlavne o AWQ, GPTQ, bitsandbytes, GGUF a FP8/KV-cache tooling

To je nejlepsi kostra pro script.
