# Jak funguje inference u LLM

Aktualizovano: 2026-03-28

Technicky zamereny script pro dospele publikum. Hlavni reference je kniha [Inference Engineering](../Downloads/Inference%20Engineering.pdf) od Philipa Kielyho, doplnena o paper-level framing kolem transformeru, attention, KV cache a modernich inferencnich technik.

## Script

### 0:00-0:45

Kdyz dnes nekdo rekne, ze pouziva LLM, ve skutecnosti skoro vzdy mluvi o inferenci, ne o trenovani. Model uz je hotovy, vahy jsou zmrazene a cilem je co nejrychleji, nejlevneji a co nejspolehliveji prevest vstupni sekvenci tokenu na dalsi token. To zni jednoduse, ale ve skutecnosti je inference kombinace numeriky, architektury modelu, prace s pameti a systemoveho inzenyrstvi. A prave proto je rozdil mezi tim, ze model existuje, a tim, ze se da dobre provozovat.

### 0:45-1:35

Zacneme od uplneho zacatku. LLM je autoregresivni model. Nedela nic mystickeho. V kazdem kroku dostane sekvenci predchozich tokenu a spocita rozdeleni pravdepodobnosti nad dalsim tokenem. Pak se podle zvoleneho rezimu dekodovani vybere jeden kandidat a cely proces se opakuje. Z pohledu runtime je zasadni, ze model negeneruje odpoved najednou. Odpoved vznika postupne, token po tokenu, a kazdy dalsi krok zavisi na vsem, co prislo predtim.

### 1:35-2:20

Jeste pred samotnym modelem probiha tokenizace. To je prakticky step zero cele inference. Text se rozbije na subword tokeny podle konkretniho tokenizeru a konkretni chat template. To je dulezite vic, nez se bezne rika. Nejen proto, ze tokenizace urcuje cenu a delku kontextu, ale i proto, ze ruzne modely maji ruzne sablony pro role, systemove prompty, tool calls a strukturovane vstupy. Pokud inference engine aplikuje template spatne, neresite jen kosmetickou chybu. Realne degradujete kvalitu.

### 2:20-3:20

Jakmile mame tokeny, prijdou embeddingy. Token je diskretni identifikator, embedding je husty vektor v prostoru o stovkach az tisicich dimenzi. Odtud vstupuje sekvence do hlavniho tela modelu, coz je serie transformer bloku. Kazdy blok ma v zasade tri dulezite casti: attention, feed-forward sit a normalizaci. Feed-forward cast obsahuje vetsinu vah modelu a je silne zalozena na matmulech. Attention je naopak inferencne komplikovanejsi, protoze propojuje aktualni token s predchozi sekvenci. Pokud chcete pochopit inference performance, musite prestat vnimat LLM jako abstraktni inteligenci a zacit ho vnimat jako dlouhou sekvenci maticovych nasobeni a presunu dat.

### 3:20-4:15

Tady se dostavame k nejdulezitejsim dvema fazim inference: prefill a decode. Prefill je pruchod celym vstupem. Model zpracuje cely prompt paralelne, spocita reprezentace pro vsechny vstupni tokeny a vytvori KV cache. Decode je potom vlastni generovani, kde uz model postupuje autoregresivne a v kazdem forward passu vyrobi dalsi token. Tohle rozliseni je klicove, protoze kazda z tech dvou fazi ma uplne jiny vykonnostni profil. Prefill obvykle urcuje time to first token. Decode urcuje tokens per second.

### 4:15-5:15

Proc jsou tyto dve faze tak odlisne? Protoze prefill je typicky compute-bound, zatimco decode byva memory-bound. V prefillu nactete vahy a provedete velke maticove operace nad delsi sekvenci. To znamena relativne hodne vypoctu na jednotku prenesenych dat, tedy vyssi arithmetic intensity. V decode je situace jina. Generujete jeden token, ale musite znovu cist velkou cast modelovych vah z pameti. Pocet operaci na jeden byte presunuty z pameti je nizsi, takze bottleneck neni cisty vypocet, ale memory bandwidth. To je jeden z hlavnich pointu knihy Inference Engineering a je to presne duvod, proc se LLM chovaji jinak v TTFT a jinak v TPS.

### 5:15-6:10

Do toho vstupuje attention. Ve standardni podobe attention porovnava query aktualniho tokenu s keys predchozich tokenu a pres values z nich sklada novy stav. Na papire attention skaluje kvadraticky se sekvencni delkou. V praxi je decode pouzitelny jen diky KV cache. Ta uklada keys a values pro predchozi tokeny, takze je nemusime pokazde pocitat znovu. Bez KV cache by autoregresivni inference byla absurdne pomala. S KV cache ale vznika novy problem: pamet. Dlouhy kontext neni drahy jen kvuli samotnym vaham. Je drahy, protoze kazdy token zvetsuje cache, ktera sedi typicky ve VRAM a soutezi o misto s vahami, aktivacemi a buffery inference enginu.

### 6:10-7:05

A tady prichazi dospela cast debaty o inferenci. Vetsina lidi se pta, jaky model je nejchytrejsi. Inference engineer se pta, jaky model se vejde do rozpoctu latence, VRAM a jednotkove ekonomiky. Z hlediska provozu je rozdil mezi 8B dense modelem, 32B dense modelem a velkym MoE modelem dramaticky. U dense modelu jsou aktivni prakticky vsechny hlavni vahy. U MoE se aktivuje jen podmnozina expertu pro kazdy token. Proto muze model typu Qwen3.5 nebo jine moderni MoE rodiny pusobit obrovsky podle celkoveho poctu parametru, ale lokalne se chovat efektivneji, pokud je pocet aktivnich parametru vyrazne nizsi nez pocet celkovy. To je dulezite hlavne pro single-request inference. V batched produkcni inferenci se tento obrazek komplikuje, protoze ruzne requesty aktivuji ruzne experty a sparsita se castecne ztraci.

### 7:05-8:00

Jak se tedy inference zrychluje v praxi? Prvni velka paka je kvantizace. Kdyz snizite presnost z BF16 nebo FP16 na FP8 nebo niz, ziskate dve veci. Compute-bound casti mohou bezet na levnejsi numerice s vyssim efektivnim vykonem a memory-bound casti presouvaji mene dat. To zni skoro jako vyhra bez kompromisu, ale neni. Ruzne casti modelu maji ruznou citlivost na kvantizaci. Linearni vrstvy byvaji relativne tolerantni. KV cache je citlivejsi. Attention, zejmena softmax a souvisejici numerika, je rizikova. Proto kvalitni inference neznamena jen model "zmensit", ale vedet presne, co kvantizovat, v jakem formatu a co ponechat ve vyssi presnosti.

### 8:00-8:45

Druha velka paka je caching a batching. Prefix caching dovoli znovu pouzit KV cache mezi requesty, pokud sdileji dostatecne dlouhy prefix. To je mimoradne dulezite pro agenty, dlouhe systemove prompty, RAG scaffolding nebo multi-turn chat. Batching zase zvysuje throughput tim, ze engine propleta vice requestu dohromady. Jenze kazda optimalizace ma tradeoff. Vyssi batch zlepsuje vyuziti GPU, ale muze poskodit individualni latenci. Prefix caching zlepsuje TTFT, ale vyzaduje cache-aware routing a rozumnou spravu pameti. V produkci tedy neexistuje jedno "nejlepsi" nastaveni. Existuje jen nastaveni vhodne pro konkretni traffic profile.

### 8:45-9:30

Kdyz se bavime o lokalni inferenci, dava smysl uvazovat mnohem pragmaticteji nez v cloud benchmarkech. Pro jednoho uzivatele nebo maly tym byva idealni model ten, ktery se vejde do dostupne VRAM nebo RAM, ma rozumny TTFT a neni tak pomaly, ze ho clovek prestane pouzivat. Prave proto dnes davaji smysl mensi dense modely a rozumne zvolene open modely z rodin jako Qwen3.5, Gemma nebo Mistral, podle konkretni ulohy. Pokud resite cestinu, lokalni naklady, soukromi a experimentovani, open modely jsou extremne zajimave. Pokud resite maximalni reasoning bez kompromisu, lokalni inference stale casto narazi na limity pameti, rychlosti a kvality.

### 9:30-10:00

Shrnuti je jednoduche. Inference neni "jen spusteni modelu". Je to optimalizace celeho retezce od tokenizace a chat template, pres transformer bloky, attention a KV cache, az po kvantizaci, batching, routing a vyber hardwaru. Pokud chcete LLM opravdu chapat, nestaci sledovat benchmarky a marketing. Musite rozumet tomu, kde se pali FLOPS, kde tece pamet a proc je prefill jiny problem nez decode. A prave tam zacina skutecne inference engineering.

## Notes

- Publikum: technicky gramotni dospeli, ne zacatecnici.
- Ton: vecny, presny, bez hype.
- Hlavni framing z knihy: runtime vrstva, memory-vs-compute bottlenecks, specialized serving.
- Doporucene papers do popisu videa:
  - Attention Is All You Need
  - FlashAttention
  - PagedAttention / vLLM
  - Medusa
  - EAGLE
