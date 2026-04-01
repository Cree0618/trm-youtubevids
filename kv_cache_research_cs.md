# KV Cache: Výzkumný Přehled

Aktualizováno: 2026-03-31

Tento dokument je výzkumně orientovaný přehled KV cache pro inferenci LLM. Je napsaný jako samostatný referenční brief, ne jako video scénář. Cílem je propojit tři vrstvy, o kterých se často mluví odděleně:

- mechaniku transformeru, která KV cache umožňuje,
- implementační detaily, díky nimž je KV cache prakticky použitelná,
- systémovou literaturu, která z KV cache dělá problém první třídy při servingu LLM.

Důraz je na veřejně dostupné vědecké články a veřejnou implementační dokumentaci. Když je nějaké tvrzení syntézou více zdrojů, a ne přímým tvrzením jednoho konkrétního zdroje, je to výslovně uvedeno.

## Co Je KV Cache

Při autoregresivní inferenci transformeru je každý nový výstupní token generován podmíněně na všech předchozích tokenech. V decoder-only transformeru potřebuje attention modul pro aktuální token přístup ke key a value tenzorům odpovídajícím dřívějším pozicím v sekvenci. Tyto tensory jsou odvozeny ze skrytých stavů dřívějších tokenů a jejich opakovaný vypocet je drahý.

`KV cache` je uložená kolekce těchto key (`K`) a value (`V`) tenzorů pro předchozí tokeny, vrstvu po vrstvě. Místo aby model při každém decode kroku znovu počítal celý attention stav nad celým prefixem, spočítá query pro aktuální pozici a attenduje nad již uloženými key a value z předchozích kroků. To je hlavní důvod, proč je autoregresivní dekódování u velkých modelů tak rychle a prakticky proveditelné.

Bez KV cache by každý nově generovaný token vyžadoval znovu spočítat reprezentace key/value pro celý prefix napříč všemi decoder vrstvami. To by dekódování dramaticky prodražilo a opakovaně by se počítaly hodnoty, která už je známe z předchozích kroků. S KV cache se hodnoty pro minulé tokeny znovu využiji; přidat je potřeba jen příspěvek aktuálního tokenu.

Je užitečné rozlišovat dvě fáze inference:

- `Prefill`: model zpracuje vstupní prompt paralelně, spočítá skryté stavy pro tokeny promptu a vytvoří počáteční KV cache.
- `Decode`: model generuje token po tokenu, čte existující KV cache, spočítá key/value pro nový token, připojí je a pokračuje dál.

Tohle rozlišení je důležité, protože KV cache vzniká hlavně během prefillu, ale při decode se z ní stává dominantní čtecí cesta.

Intuice nákladů na paměť je přímočará:

- velikost cache roste s `délkou sekvence`,
- a s `počtem vrstev`,
- a s `počtem hlav` nebo `počtem KV hlav`,
- a s `dimenzí hlavy`,
- a s `přesností`, v jaké jsou K a V uložené.

Proto může být dlouhý kontext omezen pamětí, i když se samotné váhy modelu do GPU paměti vejdou relativně pohodlně.

### Intuice

KV cache je nejlepší chápat jako způsob, jak nepočítat znovu stejný historický attention stav pořád dokola. Jakmile dřívější token jednou vyrobí své keys a values, další decode kroky je už nemusí znovu stavět; stačí je načíst, porovnat s novou query a přidat další záznam pro nově vygenerovaný token.

## Základ Transformeru a Attention

Základ transformeru pochází z práce *Attention Is All You Need* (Vaswani et al., 2017), která představila scaled dot-product attention jako klíčový primitiv architektury. Článek: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

Kompaktní tvar rovnice attention je:

\[
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Pro inferenční intuici ale není nejdůležitější samotná rovnice, nýbrž význam jednotlivých tenzorů:

- `Q` je query aktuálního tokenu nebo aktuálně zpracovávaných pozic,
- `K` je sada key reprezentujících dříve zpracované pozice,
- `V` je sada value nesoucích informaci, kterou bude attention míchat.

V decoder-only self-attention se key a value pro dřívější tokeny po jejich výpočtu v rámci daného requestu už nemění. To je strukturální důvod, proč caching funguje: jakmile je token zpracován, jeho K a V je možné znovu použít v dalších decode krocích.

Shrnutí lidskou řečí:

- každý token v každé attention vrstvě produkuje query, key a value,
- aktuální token se pomocí query ptá: „které dřívější tokeny jsou pro mě důležité?“ a porovnává je s uloženými keys,
- pak čte váženou kombinaci uložených values,
- a protože jsou dřívější keys/values pro daný request stabilní, je možné je ukládat místo jejich opakovaného přepočtu.

Proto je KV cache úzce svázaná s implementací self-attention, ne jen s modelem v abstraktní rovině.

### Intuice

Aktuální token se chová jako čtenář, který listuje rostoucím zápisníkem předchozích tokenů. Queries jsou vzor vyhledávání, keys fungují jako index a values jsou obsah, který se po nalezení shody skutečně čte. KV cache funguje právě proto, že se tento zápisník minulých tokenů nemusí v každém decode kroku přepisovat.

## Intuice Implementace

Nejjednodušší způsob, jak o implementaci KV cache přemýšlet, je tento:

- každý request vlastní rostoucí sadu uložených K a V tenzorů,
- organizovaných po vrstvách,
- které se během decode přidávají token po tokenu,
- a následné decode kroky je opakovaně čtou.

### Request-local KV cache

Základní případ je request-local caching: jeden request si vytvoří vlastní cache, během dekódování ji používá a po dokončení requestu ji zahodí. V mainstreamových autoregresivních inferenčních systémech pro LLM je nějaká forma per-request znovupoužití KV standardní provozní pattern, protože jinak by decode opakovaně přepočítával historický attention stav.

### Intuice sekvenčně orientovaného uložení

Naivní implementace ukládá cache pro každý request jako převážně souvislou sekvenci bloků nebo tenzorů indexovaných podle pozice tokenu. To je intuitivní, ale v praxi to vytváří problémy:

- requesty mají velmi různou délku,
- decode cache postupně rozšiřuje,
- dokončené requesty uvolňují paměť v nepravidelných vzorech,
- workloady s dlouhým kontextem vytvářejí velké a nerovnoměrné alokace.

Pokud systém předpokládá souvislé uložení, vzniká tlak na fragmentaci.

### Intuice blokového / stránkovaného uložení

Stránkované nebo blokově orientované uložení zachází s KV cache spíš jako s virtuální pamětí než jako s jedním obrovským souvislým polem. Místo předpokladu, že cache requestu musí být v GPU paměti fyzicky souvislá, může inference engine ukládat KV cache do bloků pevné velikosti a mapovat logické pozice sekvence na nesouvislou fyzickou paměť.

Právě tento posun je jedním z klíčových vhledů systémové literatury o servingu LLM.

### Proč je souvislá alokace problém

Souvislá alokace je konceptuálně jednoduchá, ale začne být křehká, když:

- současně běží mnoho requestů různých délek,
- některé requesty skončí dřív,
- některé requesty mají dlouhé prompty a krátké odpovědi,
- jiné mají krátké prompty a dlouhé odpovědi,
- prefix reuse potřebuje sdílet bloky cache mezi requesty.

Pokud je ukládání příliš rigidní, paměť se může plýtvat i v situaci, kdy by celková volná kapacita jinak stačila.

### Proč záleží na lokalitě, fragmentaci, reuse a transferu

KV cache ovlivňuje efektivitu servingu několika různými mechanismy:

- `Lokalita`: decode opakovaně čte uložená data; špatné přístupové vzory zhoršují výkon.
- `Fragmentace`: nepravidelné životnosti a variabilní délky plýtvají pamětí, pokud je alokace naivní.
- `Reuse`: opakující se prefixy nebo stejné prompt scaffoldingy dělají už jednou spočítanou cache cennou.
- `Transfer`: pokud jsou prefill a decode rozdělené mezi workery nebo nody, stává se cache objektem, který je potřeba efektivně přesouvat.

### Request-local cache vs sdílená / znovupoužitelná prefix cache

Tyto dvě myšlenky spolu souvisejí, ale nejsou totožné:

- `Request-local KV cache`: reuse uvnitř jednoho requestu napříč decode kroky.
- `Sdílená / znovupoužitelná prefix cache`: reuse mezi různými requesty, které sdílejí stejný prefix promptu.

Ve druhém případě už nejde jen o detail implementace transformeru; stává se z toho systémový a plánovací problém.

### Intuice

V malém měřítku vypadá KV cache jako tensorový buffer patřící jednomu requestu. Ve servingovém měřítku se ale chová spíš jako paměťový objekt, který se alokuje, sdílí, přesouvá, evikuje a někdy i znovu používá. Právě tenhle posun perspektivy spojuje mechaniku transformeru se systémovým výzkumem.

## Hlavní Výzkumné Směry

## PagedAttention / správa paměti

Problém:
Naivní alokace KV cache plýtvá GPU pamětí, protože requesty mají různou délku i životnost.

Základní myšlenka:
Ukládat KV cache do bloků pevné velikosti a mapovat logické pozice sekvence na fyzické bloky, čímž se snižuje fragmentace a zlepšuje využití paměti.

Proč to bylo důležité:
KV cache se tím posunula z role nepříjemného implementačního detailu na úroveň abstrakce pro správu paměti, která umožňuje praktický serving ve velkém měřítku.

Co zůstalo nevyřešené:
Stránkovaná správa cache řeší fragmentaci a efektivitu alokace, ale sama o sobě neřeší politiku prefix reuse, náklady přenosu při dlouhém kontextu ani disaggregated serving.

Intuice:
Nejjednodušší mentální model je virtuální paměť pro attention stav. Request vidí jednu logickou sekvenci, ale podkladové KV bloky nemusí ve fyzické GPU paměti ležet vedle sebe.

## Prefix caching / reuse cache

Problém:
Mnoho workloadů opakovaně posílá prompty s velkými sdílenými prefixy, například systémové prompty, dokumentový kontext nebo historii konverzace.

Základní myšlenka:
Znovu použít už jednou spočítanou KV cache pro sdílený prefix, takže následné requesty mohou část prefillu přeskočit.

Proč to bylo důležité:
Zlepšuje to time-to-first-token i throughput u workloadů s opakovanými prefixy, aniž by se měnily výstupy modelu.

Co zůstalo nevyřešené:
Prefix reuse pomáhá jen tam, kde se prefixy skutečně shodují; zároveň zavádí problém routingu a evikce a přímo nesnižuje decode cost.

Intuice:
Pokud mnoho requestů začíná stejným dlouhým systémovým promptem nebo stejným dokumentovým prefixem, je jeho opakovaný přepočet zbytečný. Prefix caching se snaží tuto cenu prefillu zaplatit jednou a rozprostřít ji do dalších requestů.

## Chunked prefill a scheduling

Problém:
Prefill a decode mají velmi odlišný profil využití zdrojů a velké prefillingové dávky mohou rušit průběžný decode provoz.

Základní myšlenka:
Rozdělit zpracování promptu do chunků a plánovat je tak, aby decode nestál nebo aby se lépe vyvažovala latence a throughput.

Proč to bylo důležité:
Ukázalo se, že „prefill vs decode“ není jen problém batchování, ale i problém scheduleru.

Co zůstalo nevyřešené:
Velikost chunků, plánovací politika i tail latency zůstávají závislé na konkrétním workloadu a mohou se nepříjemně potkávat s jinými optimalizacemi.

Intuice:
Velké prefillingové dávky mohou monopolizovat akcelerátor a nutit decode requesty čekat. Chunking rozseká jeden velký blok prompt práce na menší části, mezi nimiž může decode průběžně pokračovat.

## Offload / hierarchická paměť

Problém:
GPU HBM je omezená, zatímco KV cache může být u dlouhého kontextu enormní.

Základní myšlenka:
Přesouvat KV cache, nebo její chladnější části, do nižších paměťových vrstev, jako je CPU RAM nebo úložiště založené na SSD.

Proč to bylo důležité:
Rozšiřuje to proveditelnou délku kontextu a efektivní kapacitu cache nad rámec samotné GPU paměti.

Co zůstalo nevyřešené:
Cena návratu dat zpět je reálná; offload je optimalizace kapacity, ne bezplatná optimalizace latence.

Intuice:
Ne každá část cache musí neustále ležet v nejrychlejší paměťové vrstvě. Hlavní otázka zní, které KV bloky jsou dostatečně hot na to, aby si zasloužily HBM, a které mohou snést pomalejší úložiště.

## Kvantizovaná KV cache

Problém:
KV cache může při dlouhém kontextu dominovat spotřebě paměti.

Základní myšlenka:
Ukládat K a V v nižší přesnosti, čímž se zmenší footprint cache a zvýší počet tokenů, které se vejdou do paměti.

Proč to bylo důležité:
Míří to přímo na jeden z nejrychleji rostoucích spotřebitelů paměti při inferenci s dlouhým kontextem.

Co zůstalo nevyřešené:
Chyby se mohou kumulovat, protože uložené hodnoty se znovu používají v mnoha budoucích decode krocích. KV cache je citlivější než běžná komprese vah.

Intuice:
Kvantizace KV cache je výměna věrnosti reprezentace za kapacitu. Cílem není jen zmenšit tensory, ale zmenšit tu část běhového stavu, která s délkou kontextu neustále roste.

## Disaggregated serving / přenos KV cache

Problém:
Prefill a decode chtějí odlišný hardware i odlišnou scheduling strategii. Jejich spolubydlení na stejném workeru vytváří interferenci.

Základní myšlenka:
Oddělit prefill a decode na různé workery nebo instance a hotovou KV cache mezi nimi přenášet.

Proč to bylo důležité:
Umožňuje nezávisle optimalizovat prefill a decode pro TTFT a inter-token latenci.

Co zůstalo nevyřešené:
Samotný přenos KV cache se stává výkonnostním bottleneckem, zejména u dlouhého kontextu; systém tak nově závisí na kvalitě přesunu a umístění cache.

Intuice:
Jakmile se prefill a decode oddělí, přestane být KV cache čistě lokálním stavem a začne fungovat jako payload. Celý systém vyhrává jen tehdy, když je přesun tohoto payloadu levnější než interference obou fází na stejném workeru.

## Klíčové Články

## Attention Is All You Need

- Rok: 2017
- Odkaz: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

Shrnutí:
Tento článek představil transformer a mechanismus scaled dot-product attention, který umožňuje KV caching v moderní autoregresivní inferenci LLM. Neformuluje problém v termínech servingu, ale zavádí základní strukturu, kde každá pozice produkuje queries, keys a values.

Jaký problém řešil:
Nahradil rekurentní a konvoluční sekvenční modelování architekturou založenou na attention.

Proč byl důležitý pro KV cache:
KV cache je přímý provozní důsledek attention struktury transformeru.

Co neřešil:
Neřešil růst paměti při inferenci, efektivitu servingu, paging, reuse ani distribuovaný přesun cache.

## Efficient Memory Management for Large Language Model Serving with PagedAttention

- Rok: 2023
- Odkaz: [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

Shrnutí:
Práce o PagedAttention zavedla abstrakci správy paměti pro KV cache inspirovanou virtuálním stránkováním. Místo předpokladu souvislé alokace pro KV cache každého requestu ukládá cache do bloků, které mohou být ve fyzické GPU paměti nesouvislé. Hlavní přínos článku není nová attention formule, ale servingově orientované rozvržení paměti, které výrazně zlepšuje využití a throughput.

Jaký problém řešil:
Řešil plýtvání způsobené fragmentací a nadměrnou rezervací v serving systémech silně zatížených KV cache.

Proč byl důležitý:
Jde o kanonický zlom, ve kterém se z KV cache stal systémový problém první třídy v LLM servingu.

Co neřešil:
Sám o sobě neřešil politiku cross-request reuse, vzdálenější disaggregaci ani kvantizované ukládání cache.

## Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve

- Rok: 2024
- Odkaz: [https://arxiv.org/abs/2403.02310](https://arxiv.org/abs/2403.02310)

Shrnutí:
Sarathi-Serve se zaměřuje na tradeoff mezi throughputem a latencí při inferenci LLM a zavádí chunked prefill a plánování scheduleru tak, aby se omezila interference s decode. Článek chápe asymetrii mezi prefillem a decode jako systémový scheduling problém a tvrdí, že velké prefillingové iterace lze rozdělit a chytřeji prokládat s decode prací.

Jaký problém řešil:
Řešil interferenci mezi prefillem a decode a slabé využití zdrojů u smíšených workloadů.

Proč byl důležitý:
Udělal z chunked prefillu centrální koncept moderního serving designu a vyjasnil, jak scheduling souvisí se vznikem KV cache.

Co neřešil:
Neodstraňuje samotné paměťové náklady KV cache; pouze řídí, kdy a jak prefill probíhá.

## Splitwise: Efficient Generative LLM Inference Using Phase Splitting

- Rok: 2023 preprint
- Odkaz: [https://arxiv.org/abs/2311.18677](https://arxiv.org/abs/2311.18677)

Shrnutí:
Splitwise charakterizuje výpočet promptu a generování tokenů jako dvě fáze s odlišnými nároky na zdroje a navrhuje jejich rozdělení mezi samostatné stroje. Klíčový systémový vhled je, že výpočet promptu je výpočetně intenzivní, zatímco generování je více omezené pamětí, takže hardware a deployment lze optimalizovat pro každou fázi zvlášť.

Jaký problém řešil:
Řešil neefektivitu vznikající z toho, že se s prefillem a decode zachází, jako by vyžadovaly stejný profil stroje.

Proč byl důležitý:
Pomohl etablovat phase splitting jako seriózní servingový design a KV cache je zde stav, který je potřeba přesunout přes hranici obou fází.

Co neřešil:
Sám o sobě neřeší všechny inženýrské problémy spojené s velkoškálovým přenosem cache, prefix reuse ani hierarchiemi disaggregated cache.

## DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving

- Rok: 2024
- Odkaz: [https://arxiv.org/abs/2401.09670](https://arxiv.org/abs/2401.09670)

Shrnutí:
DistServe tvrdí, že společné umístění prefillu a decode svazuje alokaci zdrojů a vytváří silnou interferenci. Navrhuje jejich rozdělení a optimalizaci na goodput při dodržení service-level cílů. V této architektuře už KV cache není jen lokální stav requestu; stává se objektem, který je nutné efektivně přenášet z prefill workerů na decode workery.

Jaký problém řešil:
Řešil špatné výsledky v latenci a goodputu způsobené svázáním zdrojů mezi prefillem a decode.

Proč byl důležitý:
Udělalo z přenosu KV cache centrální systémový problém a přímo spojilo přesun cache s cíli kvality služby.

Co neřešil:
Stále závisí na efektivních interconnectech, robustních mechanismech přenosu a praktické deployment podpoře.

## Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads

- Rok: 2024
- Odkaz: [https://arxiv.org/abs/2401.11181](https://arxiv.org/abs/2401.11181)

Shrnutí:
Tento článek, často probíraný přes systém TetriInfer, tvrdí, že smíšené workloady trpí interferencí, když prefill a decode sdílejí zdroje příliš naivně. Kombinuje dělení promptů, disaggregaci a scheduling s predikcí využití zdrojů, aby zlepšil TTFT, job completion time a efektivitu na dolar.

Jaký problém řešil:
Řešil interferenci napříč různými typy requestů a heterogenními workloady.

Proč byl důležitý:
Rozšířil debatu o servingu za rámec jediného cíle typu throughput a ukázal, že serving silně závislý na KV cache musí respektovat rozmanitost workloadů.

Co neřešil:
Neodstraňuje potřebu efektivního návrhu přenosu cache ani univerzální politiky reuse.

## Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

- Rok: 2024
- Odkaz: [https://arxiv.org/abs/2407.00079](https://arxiv.org/abs/2407.00079)

Shrnutí:
Mooncake je explicitně KV-cache-centric. Odděluje clustery pro prefill a decode a používá nevyužité CPU, DRAM a SSD zdroje k vybudování disaggregated hierarchie KV cache. Článek rámuje serving s dlouhým kontextem jako problém zásadně omezený kapacitou cache a jejím přesunem a kolem toho pak staví scheduling i early rejection policy.

Jaký problém řešil:
Řešil kapacitní a plánovací bottlenecky při produkčním servingu s dlouhým kontextem a velkou cache.

Proč byl důležitý:
Je to jeden z nejjasnějších článků, které zacházejí s KV cache jako s hlavním zdrojem, kolem něhož má být postavená serving architektura.

Co neřešil:
Jeho nápady jsou obzvlášť přesvědčivé pro velké průmyslové serving systémy, ale architektura je složitější než jednodušší deployment patterny běžné ve veřejném open-source prostředí.

## KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

- Rok: 2024
- Odkaz: [https://arxiv.org/abs/2402.02750](https://arxiv.org/abs/2402.02750)

Shrnutí:
KIVI je specializovaný článek o kvantizaci KV cache. Jeho hlavní vhled je, že keys a values mají odlišné statistické vlastnosti a neměly by být nutně kvantizovány stejným způsobem. Navrhuje asymetrickou 2bitovou kvantizaci s různými strategiemi pro keys a values a ukazuje výraznou kompresi cache při lepším zachování kvality než u naivních low-bit návrhů.

Jaký problém řešil:
Řešil rostoucí dominanci KV cache ve spotřebě paměti při inferenci s dlouhým kontextem.

Proč byl důležitý:
Udělalo z kvantizace KV cache samostatné výzkumné téma, ne jen poznámku pod čarou v obecné kvantizaci modelů.

Co neřešil:
Článek sám o sobě nedokazuje, že jeho přístup je nejsnazší cesta pro integraci do obecných runtime systémů; v současných veřejných toolinzích se viditelněji prosadila jednodušší FP8 podpora KV cache, ale to je implementační trend, ne tvrzení samotného článku.

## KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization

- Rok: 2024
- Odkaz: [https://arxiv.org/abs/2401.18079](https://arxiv.org/abs/2401.18079)

Shrnutí:
KVQuant zkoumá inferenci s ultra dlouhým kontextem a tvrdí, že KV cache se velmi rychle stává dominantním bottleneckem zdrojů. Zkoumá citlivostně orientované a neuniformní schémata kvantizace KV cache s cílem dramaticky prodloužit realizovatelný kontext, zejména v režimech, kde už samotné váhy modelu nejsou hlavním problémem.

Jaký problém řešil:
Řešil problém, jak posunout délku kontextu daleko za hranici toho, co zvládne naivní full-precision KV cache.

Proč byl důležitý:
Jasně ukázal, že serving s dlouhým kontextem je často více problém kapacity cache než kapacity vah.

Co neřešil:
Stejně jako jiná agresivní práce na kvantizaci KV cache naráží na integrační složitost a tradeoffy přesnosti oproti jednodušším, produkčně přívětivějším schématům.

## Další Články, Které Materiálně Rozšiřují Pokrytí

Následující články nejsou v minimálním seznamu výše, ale významně rozšiřují systémový obrázek KV cache:

### Prompt Cache: Modular Attention Reuse for Low-Latency Inference

- Rok: 2023
- Odkaz: [https://arxiv.org/abs/2311.04934](https://arxiv.org/abs/2311.04934)

Proč ho zahrnout:
Posouvá diskusi od prostého prefix reuse směrem k modulárnímu znovupoužití attention, takže pomáhá chápat reuse cache jako něco víc než „buď shoda celého prefixu, nebo nic“.

### ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition

- Rok: 2024
- Odkaz: [https://arxiv.org/abs/2402.15220](https://arxiv.org/abs/2402.15220)

Proč ho zahrnout:
Přímo propojuje prefix-aware KV cache s chunked attention partitioningem, takže je relevantní jak pro implementaci, tak pro debatu o dlouhém kontextu a scheduling.

## Současné Implementační Patterny

Tato sekce shrnuje, co je veřejně viditelné v současných toolinzích k datu 2026-03-31. Tam, kde je tvrzení opřené o oficiální dokumentaci, jsou přiložené přímé odkazy.

### Důležité z hlediska paperů

Tyto myšlenky jsou centrální pro pochopení oboru, i když nejsou všechny stejně mainstreamové v každodenním open-source deploymentu:

- stránkovaná správa KV cache,
- prefix caching / reuse cache,
- chunked prefill,
- disaggregated prefill/decode,
- kvantizace KV cache,
- hierarchické ukládání cache mezi GPU a ne-GPU paměťovými vrstvami.

### Co je skutečně běžně vidět ve veřejných toolinzích

Tyto patterny jsou jasně přítomné ve veřejné dokumentaci a uživatelsky dostupných inferenčních frameworcích:

- `PagedAttention-style KV block management` ve vLLM, původně z práce o PagedAttention,
- `Automatic prefix caching` ve vLLM,
- `Quantized KV cache`, zejména FP8 KV cache, ve vLLM,
- `Experimental disaggregated prefilling` ve vLLM.

Relevantní oficiální dokumentace:

- vLLM Automatic Prefix Caching:
  - [https://docs.vllm.ai/en/latest/design/prefix_caching/](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- vLLM Quantized KV Cache:
  - [https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/)
- vLLM Disaggregated Prefilling:
  - [https://docs.vllm.ai/usage/disagg_prefill.html](https://docs.vllm.ai/usage/disagg_prefill.html)

Inference:
Je rozumné říct, že bloková / stránkovaná správa KV cache, prefix reuse a FP8 podpora KV cache jsou jasně viditelné ve významných veřejných toolinzích, například ve vLLM. Na základě zde citovaných zdrojů by ale bylo příliš silné tvrdit, že jde o univerzální standard napříč celým open-source ekosystémem. Stejně tak by bylo příliš silné tvrdit, že složité disaggregated hierarchie cache typu Mooncake jsou standardní součástí veřejného open-source provozu, i když jsou jejich myšlenky vlivné.

### Implementační detaily viditelné ve veřejné dokumentaci

Z dokumentace vLLM pro prefix caching:

- KV cache je spravována v blocích, ne jako jeden monolitický souvislý buffer.
- Reuse je implementováno pomocí hash-based identity bloků nad tokeny a prefix kontextem.
- Prefix caching je explicitně popsaný jako přeskočení opakovaného výpočtu promptu.

Z dokumentace vLLM pro kvantizovanou KV cache:

- FP8 KV cache je uživatelsky dostupná funkce.
- Ve veřejné dokumentaci se objevují varianty `e4m3` i `e5m2`.
- Součástí dokumentovaného workflow je i kalibrace kvantizačních scale.

Z dokumentace pro disaggregated prefill:

- veřejná dokumentace vLLM explicitně popisuje oddělené prefill a decode instance,
- a explicitně popisuje přenos KV cache mezi nimi,
- takže abstrakce přenosu KV cache je viditelná i ve veřejné implementaci, nejen v papers.

## Otevřené Problémy a Tradeoffy

### TTFT vs TPS

Optimalizace zaměřené na prefill a optimalizace zaměřené na decode často táhnou různými směry. Zlepšení time-to-first-token nemusí automaticky zlepšit tokens-per-second a naopak.

### Cache reuse vs složitost routingu

Prefix caching je velmi silné tam, kde workloady sdílejí opakující se kontext, ale funguje nejlépe tehdy, když jsou související requesty routované tam, kde už odpovídající bloky cache existují a jsou hot. To zavádí systémový problém umisťování a evikce.

### Dlouhý kontext vs růst paměti

Delší kontext dělá modely užitečnějšími, ale velikost cache roste s délkou sekvence a může zahltit HBM i v případech, kdy jsou samotné váhy modelu ještě zvládnutelné.

### Úspora díky kvantizaci vs kumulace numerické chyby

Kvantizace KV cache může odemknout výrazné úspory paměti, ale vzniklé chyby se znovu používají v mnoha budoucích krocích. Rizikový profil kvality je proto jiný než u samotné kvantizace vah.

### Přínosy disaggregace vs overhead přenosu KV cache

Oddělení prefillu a decode může zlepšit kontrolu latence i specializaci zdrojů, ale cache se tím stává objektem síťového přenosu. U dlouhého kontextu může být přesun cache natolik drahý, že část zisků zase smaže.

## Hlavní Závěry

- KV cache existuje proto, že by decoder self-attention jinak při každém decode kroku znovu počítala stejné historické key/value tensory.
- Prefill cache vytváří; decode ji opakovaně čte a rozšiřuje.
- V moderním LLM servingu bývá KV cache větší systémový bottleneck, než by člověk čekal při pohledu jen na váhy modelu.
- PagedAttention proměnilo KV cache z implementačního detailu na problém správy paměti.
- Prefix caching mění KV cache z problému reuse uvnitř requestu na problém reuse i mezi requesty.
- Chunked prefill a disaggregated serving zacházejí s KV cache jako se součástí plánovacího a přenosového pipeline.
- Kvantizace KV cache je důležitá proto, že inference s dlouhým kontextem může být dominována footprintem cache.
- Veřejné toolingy už mnoho z těchto myšlenek odrážejí, ale nejkomplexnější průmyslové architektury cache jsou zatím spíš vlivné než univerzálně standardní.

## Stručná Bibliografie

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
