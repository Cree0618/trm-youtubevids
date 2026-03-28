# Local LLM Crash Course

Aktualizovano: 2026-03-11

Detailni benchmarky a zdroje jsou v [local_llm_sources.md](local_llm_sources.md). Plan animaci a storyboard je v [script_storyboard.md](script_storyboard.md).

---

Viděl jsem moc videí o "lokální AI na Macu", která míchají dohromady tři různé věci:

1. lokální inference
2. self-hosting
3. agentic asistenty

A pak to zní, jako by to byla jedna jednoduchá věc.

Není.

Tohle video je crash course pro lidi, kteří vstupují do světa:

- lokálních LLM modelů
- Apple Silicon a local inference
- OpenClaw osobních AI agentů

Cíl není prodat hype. Cíl je dát dohromady mentální mapu, která vám ušetří týdny tápání.

## 1. První zásadní rozdíl: local model versus cloud přes API

Když používáš ChatGPT, Claude, Gemini nebo jiný webový AI produkt, ve většině případů na tvém počítači neběží samotný model. Tvůj počítač jen posílá prompt na vzdálený server a dostane odpověď zpět.

Když používáš skutečně lokální model, modelové váhy máš uložené u sebe a inference běží na tvém hardwaru. Bez internetu to pořád funguje, pokud zrovna nepoužíváš externí nástroje, web search nebo cloudový endpoint.

Tohle je důležité, protože hodně lidí řekne "jedu AI lokálně", ale ve skutečnosti mají jen lokální chat UI napojené na cloudové API.

Takže:

- `cloud AI` = model běží někde jinde
- `local inference` = model běží fakt u tebe
- `self-hosted AI app` = aplikaci hostuješ u sebe, ale model může být lokální i cloudový
- `agent` = nad modelem je orchestrace, nástroje, paměť, schvalování akcí a více kroků

To nejsou synonyma.

## 2. Co člověk získá lokální inference

První věc je `kontrola`.

Můžeš si vybrat model, quantizaci, context window, sampling, nástroje, embedding model, RAG pipeline a způsob nasazení. U cloudových služeb jsi omezený tím, co provider zveřejní a dovolí.

Druhá věc je `soukromí`.

Ne absolutní a ne magické, ale výrazně lepší než když všechno posíláš třetí straně. Pokud model běží offline a pracuje s lokálními soubory, jsi v úplně jiné situaci než u čistého API workflow.

Třetí věc je `cena při vysokém používání`.

Když AI používáš málo, API je často levnější a pohodlnější. Jakmile ale stavíš agenty, dávkové workflow nebo nad AI trávíš většinu dne, tokenová cena začne být znát. Lokální inference má vysoký vstupní cost, ale pak často platíš hlavně hardware, proud a svůj čas.

Čtvrtá věc je `nezávislost`.

Model můžeš provozovat i tam, kde nechceš nebo nemůžeš spoléhat na cizí infrastrukturu.

## 3. Co lokální inference naopak neřeší

Lokální model není automaticky chytřejší jen proto, že je lokální.

Ve většině případů bude lokální model:

- horší než top cloud model
- citlivější na prompting
- méně spolehlivý v dlouhém plánování
- horší v codingu a tool use
- náchylnější k halucinacím

To neznamená, že lokální AI nedává smysl.

Znamená to, že je potřeba přestat opakovat internetový folklór typu:

- "tohle je skoro jako top proprietary model"
- "tenhle quant je prakticky bez kompromisů"
- "agent framework z malého modelu udělá génia"

Neudělá.

Framework umí zlepšit workflow. Neumí magicky dodat inteligenci, která v modelu není.

## 4. Tři vrstvy, které si musíš v hlavě oddělit

Když začínáš, pomůže jednoduchý model:

`model -> runtime -> agent system`

`Model` je samotný mozek.

Příklad:

- Qwen 3.5
- MiniMax M2.5
- Kimi K2.5
- GLM 4.7 Flash

`Runtime` je to, co model obslouží.

Příklad:

- [Ollama](https://ollama.com/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [LM Studio](https://lmstudio.ai/docs/app)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)

`Agent system` je vrstva nad tím.

Ta řeší:

- nástroje
- routing
- approvals
- paměť
- více kroků
- komunikaci přes chaty nebo dashboard

Příklad:

- [OpenClaw](https://docs.openclaw.ai/)

Hodně zmatku vzniká tím, že někdo o agentovi mluví, jako by to byl model. Není.

## 5. Co znamená "utáhne to model"

Tohle je jedna z nejčastějších nejasností.

Když někdo řekne, že jeho počítač "utáhne 70B model", může tím myslet několik úplně jiných věcí:

1. model se vejde do paměti
2. model se spustí, ale bude pomalý
3. model bude použitelný na single-user chat
4. model poběží i s delším kontextem
5. model poběží v quantizaci, která ještě drží kvalitu

To nejsou stejné věci.

U lokální inference sleduj hlavně:

- `RAM / unified memory capacity`
- `memory bandwidth`
- `quantizaci`
- `context window`
- `KV cache`
- `tokens per second`
- `time to first token`

Lidi se často soustředí jen na velikost modelu v B parametrech. To nestačí.

## 6. Nejkratší vysvětlení nejdůležitějších pojmů

### Parametry a aktivní parametry

U dense modelu se typicky používá skoro všechno.

U MoE modelu je důležité nejen kolik má model celkem parametrů, ale kolik jich aktivuje na jeden token. Proto může mít jeden model 400B total params, ale běžet rychleji než jiný menší dense model.

Pro rychlost decode je často důležitější `active params` než `total params`.

### Quantizace

Quantizace znamená, že se model uloží v nižší numerické přesnosti, aby zabíral méně paměti a často běžel rychleji.

Typicky platí:

- nižší bitová šířka = menší footprint
- nižší bitová šířka = často vyšší rychlost
- nižší bitová šířka = větší riziko ztráty kvality

Ne každá 4bit quantizace je stejně dobrá. To je důvod, proč dvě videa o "tom samém modelu" můžou končit úplně jiným dojmem.

### Context window a KV cache

Context window je množství textu, které model vidí najednou.

Jenže větší kontext není zdarma. Roste KV cache, tedy další paměť potřebná během inference. Takže model může "na papíře běžet", ale při velkém kontextu být najednou nepříjemně pomalý nebo dokonce nepraktický.

### Latence versus throughput

`Latence` je, jak rychle dostane odpověď jeden uživatel.

`Throughput` je, kolik práce systém udělá celkem.

Dva počítače ti často zvednou throughput víc než subjektivní pocit "to píše výrazně rychleji". To je důležité hlavně u clusterů a agentických workflow.

## 7. Proč jsou Apple Silicon stroje pro local LLM zajímavé

Apple Silicon dává smysl hlavně kvůli `unified memory`.

Na klasickém PC s NVIDIA GPU jsi velmi rychle omezený VRAM. Na Apple Silicon sdílí CPU a GPU jednu paměť. To je pro inference velkých modelů prakticky velmi důležité.

Druhá věc je `memory bandwidth`.

Právě ta často limituje rychlost generování. U LLM inference je bottleneck často paměť, ne marketingové FLOPS.

Třetí věc je `pohodlí`.

Apple Silicon stroje jsou tiché, relativně úsporné a na local inference často méně otravné než vlastní CUDA workstation.

To je reálný důvod, proč se kolem nich udělal hype.

## 8. Velká oprava internetového memu: Mac mini není Mac Studio

Tohle je potřeba říct natvrdo.

Spousta videí míchá dohromady `Mac mini` a `Mac Studio`, jako by to byla jen menší a větší verze téhož.

Není.

Aktuální [Apple Mac mini specs](https://www.apple.com/mac-mini/specs/) uvádí:

- `M4`: paměťová propustnost `120 GB/s`
- `M4 Pro`: paměťová propustnost `273 GB/s`
- Mac mini končí na `64 GB` unified memory

Aktuální [Mac Studio specs](https://www.apple.com/mac-studio/specs/) a [M3 Ultra announcement](https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/) uvádí:

- až `512 GB` unified memory
- `819 GB/s` memory bandwidth
- `Thunderbolt 5`

To je jiná liga.

Když někdo mluví o provozu modelů typu MiniMax M2.5, Qwen3.5-397B nebo Kimi K2.5 na "Macu", musíš se okamžitě ptát:

- Je to Mac mini?
- Je to Mac Studio?
- Kolik má RAM?
- Jaká je quantizace?
- Jaká je rychlost?

Bez toho je to jen marketingový dojem.

## 9. Co realisticky čekat od Mac mini a Mac Studio

Jako hrubá mentální mapa:

- `Mac mini M4` je dobrý na menší modely, lehčí chat, shrnování, dokumenty a experimenty
- `Mac mini M4 Pro` je zajímavý pro serióznější local práci, ale pořád to není stroj na největší open modely
- `Mac Studio M3 Ultra 512 GB` je stroj, na kterém se otevírá svět opravdu velkých open modelů

Prakticky:

- na Mac mini dává smysl myslet v kategoriích menších dense modelů a menších nebo rychlých MoE
- na Mac Studio dává smysl uvažovat i o MiniMax M2.5, větších Qwen 3.5 modelech nebo Kimi K2.5 quantech

Takže ano: Apple Silicon hype má reálný základ.

Ale ne: neznamená to, že každá malá Apple krabička je najednou náhrada za velký GPU server.

## 10. Dnešní local stack pro normálního člověka

Pokud jsi začátečník, nechceš řešit všechno od nuly. Dnešní praktický stack vypadá zhruba takto:

### Nejjednodušší start

- [LM Studio](https://lmstudio.ai/docs/app)
- [Ollama](https://ollama.com/)

Tohle je ideální, pokud chceš:

- stáhnout model
- poslat prompt
- otestovat rychlost
- porovnat dva modely vedle sebe

### Univerzální low-level runtime

- [llama.cpp](https://github.com/ggml-org/llama.cpp)

Tohle je dobré, když chceš:

- maximum kontroly
- server mode
- OpenAI-compatible endpoint
- benchmarkovat a ladit

### Apple Silicon heavy-duty cesta

- [MLX-LM](https://github.com/ml-explore/mlx-lm)

Tohle je dnes nejdůležitější runtime pro opravdu velké modely na Apple Silicon, zejména pokud chceš:

- MLX quants
- nejlepší praktické výsledky na velkých modelech
- experimenty s Apple-native ekosystémem

### Frontend a self-hosted UI

- [Open WebUI](https://docs.openwebui.com/)

Tohle je dobré, pokud chceš:

- chat UI
- více providerů
- pipelines
- RAG
- tools
- lokální i cloud modely vedle sebe

Moje praktická rada:

1. začni v LM Studio nebo Ollama
2. pochop modely, quants a rychlost
3. teprve potom řeš RAG
4. a až pak agent systems

## 11. Jak dnes vybírat modely

Když jsi nový, přestaň se ptát jen "kolik B model utáhne můj počítač".

Správnější otázky jsou:

- na co ten model chci
- jak rychle má odpovídat
- jestli chci spíš coding, research nebo všeobecný chat
- kolik mám RAM
- jestli potřebuji multimodalitu
- jestli potřebuji tool use a structured outputs

### Pro rychlost a každodenní použití

Modely typu:

- menší dense modely
- menší MoE s nízkým active-param footprintem

budou často subjektivně příjemnější než obrovský frontier model, který se sice vejde, ale každá odpověď trvá moc dlouho.

### Pro seriózní local coding a research

Dnes dávají smysl hlavně:

- [Qwen 3.5 family](https://qwen.ai/blog?id=qwen3.5)
- [MiniMax M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5)
- [Kimi K2.5](https://huggingface.co/moonshotai/Kimi-K2.5)
- [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)

Krátká orientace:

- `Qwen3.5-35B-A3B` je velmi zajímavý na rychlost
- `Qwen3.5-122B-A10B` je skvělý kompromis velikost versus kvalita
- `Qwen3.5-397B-A17B` je jeden z nejsilnějších open all-round modelů
- `MiniMax M2.5` je extrémně zajímavý pro coding a search-heavy úlohy
- `Kimi K2.5` je velmi silný multimodální agentic model, ale je lokálně náročný

Detailní benchmarky jsou v [local_llm_sources.md](local_llm_sources.md).

## 12. Co dnes platí pro speed a kvalitu na Apple Silicon

Hrubé pravidlo:

- čím víc active params na token, tím nižší rychlost decode
- čím větší quant, tím víc paměti a často nižší rychlost
- čím delší context, tím větší tlak na KV cache

To je důvod, proč model může být:

- výborný na papíře
- použitelný v krátkém benchmarku
- ale nepříjemný v reálném workflow

Právě proto má smysl u lokálních modelů vždy uvádět:

- model
- quant
- RAM
- context
- tok/s

Bez toho je to šum.

## 13. OpenClaw: co to je doopravdy

[OpenClaw docs](https://docs.openclaw.ai/) popisují OpenClaw jako self-hosted AI gateway a osobní agent systém, ne jako "launcher lokálních modelů".

To je správný framing.

OpenClaw řeší:

- zprávy a komunikační kanály
- dashboard
- identity a pairing
- schvalování akcí
- workspaces
- nástroje
- routování na různé modely
- možnost běžet lokálně i přes API

Jinými slovy:

OpenClaw není hlavně o tom "mít model doma".

Je to o tom mít vlastního asistenta pod svou kontrolou.

## 14. Co je v OpenClaw dnes nové a důležité

Aktuální dokumentace ukazuje onboarding přes:

- [doporučený install script](https://docs.openclaw.ai/quickstart), dnes `curl -fsSL https://openclaw.ai/install.sh | bash`
- `openclaw onboard --install-daemon`
- `openclaw dashboard`
- `openclaw gateway status`

Aktuální dashboard docs ukazují browser-based Control UI pro:

- active tasks
- approvals
- agent routing
- workspaces
- connected channels

To je důležité, protože starší mentální model "je to jen chat bot v nějakém messengeru" už nestačí.

Quickstart navíc explicitně počítá s `Node 22+`, což je malý detail, ale přesně ten typ detailu, na kterém se setup zbytečně láme.

Aktuální docs také ukazují:

- [pairing model pro externí identity a zařízení](https://docs.openclaw.ai/concepts/pairing)
- [approval flow](https://docs.openclaw.ai/concepts/approvals)
- [sandboxing a workstation model](https://docs.openclaw.ai/concepts/workstation)

To je přesně ta vrstva, která odlišuje agenta od obyčejného chat UI.

## 15. Nejdůležitější pravda o OpenClaw local mode

[OpenClaw local models docs](https://docs.openclaw.ai/gateway/local-models) jsou v tomhle překvapivě střízlivé.

Ta dokumentace neříká "vezmi malý lokální model a jsi hotový". Naopak říká přibližně tohle:

- pro vážnější agent use cases chceš silný model
- dlouhý kontext je důležitý
- prompt injection je reálný problém
- lokální modely jsou možné, ale nejsou automaticky nejlepší volba

Docs dnes jako lokální setup ukazují [LM Studio](https://lmstudio.ai/docs/app) a full-size [MiniMax M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) přes Responses API. To je důležité číst správně:

- workflow doporučení je aktuální
- konkrétní model v docs nemusí být vždy nejnovější frontier release

Moje inference z aktuálních model cards je:

- pokud chceš čistě lokální high-end na Apple Silicon, dnes dávají větší smysl i novější modely typu MiniMax M2.5 nebo větší Qwen 3.5
- ale samotné doporučení "nešetři příliš na modelu, pokud chceš agenta" je pořád přesné

Tohle je inference z více zdrojů, ne přímá citace OpenClaw docs.

## 16. Proč agent potřebuje lepší model než obyčejný chat

Když model používáš jen jako chat, stačí, aby napsal přiměřenou odpověď.

Když ho používáš jako agenta, musí navíc:

- zvolit správný další krok
- vybrat nástroj
- správně zavolat tool
- pochopit výsledek
- odolat prompt injection v datech nebo tool outputs
- držet cíl přes více kroků

To dramaticky zvyšuje nároky na modelovou kvalitu.

Model, který je "v pohodě na chat", může být mizerný agent.

Tohle je důvod, proč tolik demo videí funguje v 30 sekundách a pak se v normálním provozu rozpadne.

## 17. Prompt injection a bezpečnost nejsou detail

Jakmile agent čte web, e-maily, poznámky nebo výstupy nástrojů, otevíráš útokovou plochu.

To je důvod, proč OpenClaw řeší:

- approvals
- pairing
- sandboxing
- oddělené workspaces

Pokud chceš dělat osobního asistenta, nestačí řešit jen "jaký model je nejchytřejší". Musíš řešit i:

- co všechno může model číst
- co všechno může model udělat
- co vyžaduje potvrzení
- co běží v sandboxu

Pro nové lidi je důležité pravidlo:

čím víc agent umí dělat, tím míň si můžeš dovolit slabý model a tím víc musíš řešit bezpečnost.

## 18. Kdy OpenClaw dává smysl

OpenClaw dává smysl, když chceš:

- self-hosted osobního asistenta
- propojit AI s komunikačními kanály
- kombinovat cloud a local modely podle typu úlohy
- mít approvals a routing pod kontrolou
- postupně budovat osobní agent stack

Nedává smysl, když si jen chceš:

- občas popovídat s modelem
- srovnat dva modely
- otestovat lokální inference poprvé

Na to je lepší začít v LM Studio, Ollama nebo Open WebUI.

OpenClaw je vrstva výš.

## 19. Dnešní realistický postup pro začátečníka

Pokud jsi nový, tohle je rozumné pořadí:

1. pusť jeden lokální model
2. nauč se, co je quant, context a tok/s
3. porovnej rychlý model a chytrý model
4. otestuj structured outputs a tool calling
5. přidej embeddings a jednoduchý RAG
6. přidej OpenAI-compatible endpoint
7. teprve potom řeš agenty a OpenClaw

Tohle pořadí má důvod.

Když přeskočíš základy, nebudeš vědět, jestli problém způsobuje:

- model
- quant
- runtime
- retrieval
- prompt
- tool orchestration
- nebo bezpečnostní restrikce

## 20. Kdy má smysl cloud, local a hybrid

`Cloud` dává smysl, když chceš:

- maximální kvalitu
- nejlepší coding a planning
- minimum setupu

`Local` dává smysl, když chceš:

- soukromí
- kontrolu
- offline nebo low-trust workflow
- levnější dlouhodobý provoz

`Hybrid` dává smysl nejčastěji.

Tohle je mimochodem nejstřízlivější setup i pro OpenClaw:

- lokálně jednoduché nebo citlivé věci
- v cloudu nejtěžší reasoning a plánování

To je obvykle realističtější než představa, že všechno poběží lokálně a bez kompromisu.

## 21. Mac versus NVIDIA workstation

Tady neexistuje univerzálně správná odpověď.

`Apple Silicon` je skvělý, když chceš:

- tichý stroj
- unified memory
- jednoduchost
- single-user inference
- velké modely bez klasické VRAM pasti

`NVIDIA workstation` je lepší, když chceš:

- CUDA ekosystém
- širší kompatibilitu s experimentálními projekty
- fine-tuning
- upgrade path
- vysoký throughput pro více lidí

Mac je často lepší appliance.

NVIDIA PC je často lepší dílna.

## 22. Největší bullshit, který kolem local LLM pořád lítá

- "už nikdy nebudeš potřebovat cloud AI"
- "tenhle malý quant je skoro stejně dobrý jako top API model"
- "když mi běží 70B, mám enterprise úroveň"
- "agent framework vyřeší slabý model"
- "lokální automaticky znamená bezpečné"

Většina těch vět je buď nepravda, nebo hrubé zjednodušení.

Správnější verze je:

- pro některé use cases už nepotřebuješ cloud pořád
- některé quants jsou překvapivě dobré, ale stále jsou to kompromisy
- běžet neznamená být pohodlně použitelný
- framework zlepší workflow, ne inteligenci
- lokální setup může být soukromější, ale agentický systém přidává nové útokové plochy

## 23. Praktická doporučení podle typu člověka

Pokud jsi nový a nechceš se utopit:

- začni s LM Studio nebo Ollama
- sleduj tok/s, ne jen velikost modelu
- drž se jednoho UI a jednoho runtime, dokud chápeš základy
- neřeš agenty první den

Pokud chceš vážně lokální AI na Apple Silicon:

- řeš RAM a bandwidth dřív než SSD
- zajímej se o MLX modely a MLX-LM
- sleduj community benchmarky, ale odděluj je od vendor benchmarků

Pokud chceš osobního AI asistenta:

- nauč se approvals
- používej sandbox
- začni s úzkým use casem
- nechtěj po malém modelu, aby dělal high-stakes autonomii

## 24. Nejkratší závěr

Lokální inference je reálná, užitečná a pro spoustu lidí dává smysl.

Apple Silicon hype není vycucaný z prstu. Unified memory a bandwidth jsou pro local LLM opravdu důležité.

Ale stejně tak platí:

- Mac mini není Mac Studio
- model, který se vejde, ještě nemusí být příjemně použitelný
- agent není totéž co chat
- a OpenClaw není jen "nějaká appka pro lokální model"

Pokud chceš nejlepší mozek, stále často vyhraje cloud.

Pokud chceš kontrolu, soukromí a vlastní infrastrukturu, local a hybrid setup dávají obrovský smysl.

Pokud chceš osobního AI agenta, začni méně ambiciózně, než říká YouTube.

To je většinou nejlepší způsob, jak se k opravdu užitečnému setupu dostat rychleji.

---

## Rychlé shrnutí

- `local inference` není totéž co cloud API
- `self-hosted` není totéž co lokální model
- `agent` není totéž co model ani chat UI
- sleduj `RAM`, `bandwidth`, `quant`, `context`, `KV cache` a `tok/s`
- Mac mini a Mac Studio jsou pro local LLM dvě různé třídy hardwaru
- začátečník má začít v `LM Studio` nebo `Ollama`, ne v plném agent stacku
- OpenClaw je self-hosted agent system s approvals, pairingem a sandboxingem
- pro vážnější agenty chceš lepší model, ne menší hype
- nejrealističtější setup pro hodně lidí je `hybrid`, ne čistě local ani čistě cloud

## Odkazy

- Apple Mac mini specs: https://www.apple.com/mac-mini/specs/
- Apple Mac Studio specs: https://www.apple.com/mac-studio/specs/
- Apple M3 Ultra announcement: https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/
- Ollama: https://ollama.com/
- llama.cpp: https://github.com/ggml-org/llama.cpp
- LM Studio docs: https://lmstudio.ai/docs/app
- MLX-LM: https://github.com/ml-explore/mlx-lm
- Open WebUI docs: https://docs.openwebui.com/
- OpenClaw docs: https://docs.openclaw.ai/
- OpenClaw quickstart: https://docs.openclaw.ai/quickstart
- OpenClaw dashboard: https://docs.openclaw.ai/dashboard
- OpenClaw pairing: https://docs.openclaw.ai/concepts/pairing
- OpenClaw approvals: https://docs.openclaw.ai/concepts/approvals
- OpenClaw workstation: https://docs.openclaw.ai/concepts/workstation
- OpenClaw local models: https://docs.openclaw.ai/gateway/local-models
- Qwen 3.5 blog: https://qwen.ai/blog?id=qwen3.5
- Qwen3.5-397B-A17B: https://huggingface.co/Qwen/Qwen3.5-397B-A17B
- Qwen3.5-122B-A10B: https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8
- MiniMax M2.5: https://huggingface.co/MiniMaxAI/MiniMax-M2.5
- Kimi K2.5: https://huggingface.co/moonshotai/Kimi-K2.5
- GLM-4.7-Flash: https://huggingface.co/zai-org/GLM-4.7-Flash
