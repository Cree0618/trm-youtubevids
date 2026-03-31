# Jak funguje kvantizace u LLM

Aktualizovano: 2026-03-28

Technicky script pro 10-12 minutove video v cestine bez zbytecne popularizacni omacky. Vychazi z [quantization_research.md](./quantization_research.md) a je stavene jako hybrid:

- nejdriv prakticky deployment a inference framing
- potom paper families a jejich technicka pointa

## Script

### 0:00-0:45

Kdyz nekdo rekne, ze "ten model je ve 4 bitech", zni to skoro trivialne. Jenze pod tim jednim cislem se ve skutecnosti schovava skoro cely technicky problem. Kvantizace neni jedno tlacitko, kterym model zmensite. Je to cela trida metod, ktere meni zpusob reprezentace vah, aktivaci nebo KV cache tak, aby inference zabrala mene pameti, kladla mensi naroky na pametovou propustnost a casto byla i rychlejsi. A prave proto muze byt jeden 4-bit model vyborny a druhy skoro nepouzitelny.

### 0:45-1:35

Z pohledu inference engineeringu je duvod kvantizace celkem prizemni. LLM nejsou drahe jen proto, ze maji hodne parametru. Drahe jsou hlavne proto, ze pri inferenci porad dokola nacitaji vahy a pracuji s velkym objemem dat v pameti. A prave decode faze byva velmi casto omezena pametovou propustnosti, ne cistym vypocetnim vykonem. Kdyz tedy snizim presnost vah nebo cache, model se casto nezrychli proto, ze by pocital mene, ale proto, ze pro kazdy krok potrebuje nacist a presunout mene dat.

### 1:35-2:20

Je proto uzitecne rozdelit inference na dve faze. Prefill zpracuje cely prompt a byva casto vic compute-bound. Decode pak generuje token po tokenu a byva spis memory-bound. Kvantizace tedy nema vzdy stejny efekt. Weight-only quantization casto pomuze hlavne decode, protoze zmensi objem vah, ktere je treba neustale nacitat. Naopak kvantizace vah i aktivaci muze pomoct i prefillu, pokud pro ni ma hardware i runtime dobre optimalizovanou nizkopresnou cestu.

### 2:20-3:10

Jak kvantizace funguje mechanicky? V nejjednodussi variante vezmete floating-point hodnoty a prevedete je do mensi mnoziny reprezentovatelnych hodnot. K tomu potrebujete scale, nekdy zero-point a u nekterych metod i codebook. A presne tady se z jednoduche myslenky stava skutecny inzenyrsky problem. Jedna skala pro cely tensor je levna, ale nepresna. Per-channel nebo per-group skaly obvykle zlepsi kvalitu, ale pridaji slozitost. A pokud misto obycejne scalar quantization pouzijete vektorove kodovani nebo codebooky, dostanete se na velmi nizke bity, ale runtime uz je podstatne narocnejsi.

### 3:10-4:00

Dalsi dulezita vec je, ze ruzne casti modelu snaseji nizkou presnost ruzne dobre. Linearni vahy jsou relativne vdecne. Aktivace jsou horsi, protoze zaviseji na konkretnich datech a casto v nich vznikaji outliery. KV cache je jeste citlivejsi, protoze se nepouzije jednou, ale znovu a znovu v dalsich krocich decode. A attention samotna, hlavne vse kolem score, normalizace a vahovani kontextu, je numericky velmi citliva. Proto neni stejne rict "kvantizovali jsme vahy" a "kvantizovali jsme celou inferencni cestu vcetne aktivaci a KV cache".

### 4:00-4:55

Historicky prvni opravdu zasadni paper v moderni LLM kvantizaci je LLM.int8. Jeho pointa nebyla jen v tom, ze int8 muze fungovat. Dulezite bylo pochopeni, ze v transformerech existuji outlier dimensions, ktere rozbiji naivni kvantizaci. LLM.int8 proto zavadi mixed-precision rozklad: vetsina vypoctu bezi v int8, ale outlierove dimenze zustavaji ve vyssi presnosti. A to je duvod, proc je ten paper porad relevantni. Ukazal, ze problem kvantizace neni jen pocet bitu, ale hlavne rozdeleni hodnot.

### 4:55-5:55

Pak prisla era weight-only PTQ metod, ktera je dodnes z praktickeho hlediska nejdulezitejsi pro open-source inference. GPTQ pouziva aproximaci druheho radu, aby post-training quantization co nejmene poskodila chovani modelu. AWQ sla jinou cestou: vsimla si, ze mala cast salient vah ma neumerne velky vliv na tok aktivaci, a proto je potreba ji chranit. V realnem deploymentu je tohle porad jadro sveta kolem 4-bit checkpointu. Kdyz dnes nekdo spousti open model v nizke presnosti, velmi casto se pohybuje nekde mezi GPTQ, AWQ, bitsandbytes a formaty jako GGUF.

### 5:55-6:45

SmoothQuant je jina vetev. Neni to hlavni local-inference folklore, ale paperove je velmi dulezity, protoze otevrel cestu k W8A8 kvantizaci vah i aktivaci. Jeho hlavni myslenka je elegantni: obtiznost kvantizace aktivaci se offline presune do vah algebraicky ekvivalentni transformaci. A to je podstatne pro pochopeni, proc activation quantization neni jen "to same co kvantizace vah, ale na jinem tensoru". Aktivace maji jinou distribuci, casteji obsahuji outliery a jsou citlivejsi na ztratu presnosti.

### 6:45-7:35

Jak se obor posouval dal, zacalo byt jasne, ze pro ultra-low-bit rezimy nestaci jen chytre rounding a group scales. Objevily se metody jako AQLM a QuIP#, ktere se na kvantizaci divaji spis jako na problem vektoroveho kodovani a codebooku. Pak prisly rotational methods, hlavne QuaRot a SpinQuant. Jejich pointa je dost zasadni: mozna neni nejvetsi problem samotna kvantizace, ale to, v jakem souradnem systemu tensor kvantizujeme. Kdyz vhodne otocite reprezentaci a odstranite outliery bez zmeny chovani modelu v puvodni presnosti, kvantizace je najednou mnohem snazsi.

### 7:35-8:20

I proto bych byl opatrny u vet typu "nejlepsi quantizace je X". V roce 2026 je ten landscape dost roztristeny. Nektere metody jsou paperove velmi dulezite, ale nejsou vychozi volbou pro bezny deployment. Jine zase nejsou tak atraktivni z akademickeho hlediska, ale v praxi je potkavate porad. HQQ je dobry priklad: je calibration-free, velmi rychle, flexibilni a umi i velmi nizke bity, ale neni to univerzalni mainstream format typu AWQ checkpointu na Hugging Face. AutoRound je naopak priklad, jak se vyzkum kolem optimalniho roundingu postupne meni v produkcnejsi tooling.

### 8:20-9:10

Ted to nejdulezitejsi pro dlouhy kontext: KV cache. Spousta lidi se na kvantizaci diva jen pres vahy, jenze u dlouheho kontextu muze byt hlavni problem prave KV cache. Ta roste s poctem tokenu a pri decode se neustale cte. Proto vznikly specializovane prace jako KIVI a KVQuant. KIVI ukazuje, ze keys a values maji odlisnou distribuci a nedava smysl kvantizovat je stejne. KVQuant jde jeste dal a ukazuje, ze u opravdu dlouheho kontextu se KV cache muze stat dominantnim pametovym bottleneckem driv, nez narazite na samotne vahy modelu.

### 9:10-9:55

A tady je dulezita prakticka pointa. To, co je paperove nejzajimavejsi, neni vzdy to, co se nejvic nasazuje. V produkcnim stacku dnes casto vitezi jednodussi a hardwarove vstricnejsi reseni. Ve vLLM je typickym prikladem FP8 KV cache. Nemusi byt tak akademicky ambiciozni jako asymetricka 2-bit specializace ala KIVI, ale mnohem snaz se integruje do realneho inference enginu, zvysuje efektivni kapacitu cache, zlepsuje propustnost a dobre zapada do modernich nizkopresnych GPU pipeline.

### 9:55-10:40

Do celeho obrazku patri i QLoRA, i kdyz to neni cisty inference paper. QLoRA je dulezita proto, ze ze 4-bit sveta udelala nejen deployment trik, ale i standardni fine-tuning workflow. NF4, double quantization a frozen backbone v nizke presnosti s LoRA adaptery ukazaly, ze nizka precision neni jen otazka servingu, ale i toho, kdo si muze dovolit model vubec upravovat. Do videa bych to dal jako kratkou odbocku: ne jako hlavni osu, ale jako vysvetleni, proc ma 4-bit ekosystem dnes takovou trakci.

### 10:40-11:35

Takze co se dnes skutecne pouziva? Pro local inference jsou stale nejbeznejsi bitsandbytes, AWQ, GPTQ a GGUF-style kvantizace v runtimich jako llama.cpp. Pro serverovou inference dava stale vetsi smysl premyslet i v kategoriich FP8 a kvantizovane KV cache. A pokud nekdo tvrdi, ze "tenhle 2-bit model je bez kompromisu", je skoro jiste, ze skryva nejaky podstatny detail: bud eval setup, nebo runtime overhead, nebo to, ze sice model zabira malo, ale skutecna inferencni cesta uz tak vyhodna neni.

### 11:35-12:00

Shrnuti je jednoduche. Kvantizace neni jedno cislo v nazvu checkpointu. Je to rozhodnuti o tom, co presne kvantizujete, s jakou granularitou, v jakem formatu, nad jakym kernel stackem a s jak velkou toleranci k degradaci kvality. Pokud tomu chcete rozumet opravdu technicky, nestaci vedet, ze model je "ve 4 bitech". Musite vedet, jestli mluvite o vahach, aktivacich, KV cache nebo o cele inferencni pipeline.

## Notes

- Publikum: dospeli, technicky gramotni divaci.
- Ton: vecny, presny, bez hype.
- Script schvalne nepretizuje benchmarky; osu tvori mechanismus a tradeoff.
- Hlavni paper families:
  - LLM.int8
  - GPTQ
  - AWQ
  - SmoothQuant
  - QLoRA
  - HQQ
  - AQLM
  - QuIP#
  - AutoRound
  - QuaRot
  - SpinQuant
  - KIVI
  - KVQuant
  - QQQ
- Odpor proti marketingu:
  - "4-bit" samo o sobe nic negarantuje
  - ne kazda kvantizace je inference speedup
  - memory footprint a end-to-end latency nejsou totiz stejne metriky
