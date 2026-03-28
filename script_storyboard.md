# Manim Storyboard

Aktualizovano: 2026-03-11

Tento storyboard je přepsaný pro animace v `Manim Community`.
Nejde o přímou imitaci konkrétního autora. Vizuální směr je:

- čistý matematicko-edukační explainer
- tmavé pozadí
- vysoký kontrast
- geometrické layouty
- plynulé transformace mezi myšlenkami
- minimum dekorace, maximum srozumitelnosti

Tedy estetika vhodná pro technické video ve stylu "vizuální myšlení", ne generický motion graphics template.

## 1. Globální vizuální jazyk

### Pozadí

- tmavé modro-černé pozadí
- jemný radiální gradient nebo velmi subtilní noise texture
- žádné neon AI pozadí

### Barvy

- primární text: `WHITE`
- sekundární text: `GREY_B`
- akcent hardware: `BLUE_C`
- akcent modely: `TEAL_C`
- akcent agenti / OpenClaw: `GOLD_C`
- akcent riziko / prompt injection: `RED_C`
- akcent správná mentální mapa: `GREEN_C`

### Typografie

- běžný text přes `Text(...)`
- technické labely a symboly přes `Tex(...)` nebo `MathTex(...)`
- konzistentní menší tracking, dost whitespace
- titulky maximálně 1 až 2 řádky

### Pohyb

- používat hlavně `FadeIn`, `FadeOut`, `Transform`, `ReplacementTransform`, `LaggedStart`, `GrowArrow`, `Create`, `Write`
- minimálně používat agresivní zoomy
- radši plynulé přeuspořádání objektů než tvrdé střihy

### Kamera

- defaultně statická
- pomalé `self.camera.frame.animate.move_to(...)` jen tam, kde pomůže fokus
- žádné zbytečné cinematic nájezdy

## 2. Reusable Manim komponenty

Tyto prvky se vyplatí udělat jako helper funkce nebo malé třídy:

- `TitleBlock(title, subtitle=None)`
- `PillLabel(text, color)`
- `MetricCard(title, value, footer=None)`
- `SystemBox(label, color, icon=None)`
- `ArrowFlow(source, target, label=None)`
- `ProsConsScale()`
- `DecisionTreeNode(label)`
- `HardwareLane(title, subtitle=None)`
- `WarningStamp(text="MYTH" nebo "WRONG")`
- `PromptBubble(text)`
- `TerminalBlock(lines=[...])`
- `DashboardPanel(title)`

## 3. Obecná režie scén

Každá scéna by měla mít tento rytmus:

1. titul nebo centrální claim
2. vizuální konstrukce problému
3. jedna hlavní transformace
4. krátký závěr nebo takeaway label

Každá scéna musí mít jednu dominantní pointu. Pokud jsou na obrazovce dvě myšlenky najednou, druhá má být potlačená opacity nebo velikostí.

## 4. Scéna 1: Úvod

### Cíl

Rozdělit tři pojmy, které internet běžně míchá.

### Manim objekty

- `TitleBlock("Local LLM Crash Course")`
- tři `PillLabel`: `local inference`, `self-hosting`, `agents`
- tři velké kruhy nebo rounded rectangles

### Animace

1. `Write(title)`
2. tři labely přiletí do stejného středu pomocí `LaggedStart(FadeIn(..., shift=...))`
3. překryjí se
4. přes `animate.arrange(RIGHT, buff=1.0)` se od sebe oddělí
5. pod každý se dopíše krátká definice

### On-screen takeaway

`Tyhle 3 věci nejsou totéž`

### Implementační poznámka

Použít `VGroup` a transformovat jeden cluster do tří-column layoutu.

## 5. Scéna 2: Cloud vs local

### Cíl

Vysvětlit úplný základ.

### Manim objekty

- dva notebook-like boxy
- vlevo cloud ikonka
- vpravo box `model weights`
- šipky pro tok promptu a odpovědi

### Animace

1. vytvořit split layout
2. vlevo prompt bubble putuje do cloudu a vrací se answer bubble
3. vpravo prompt bubble vstoupí přímo do lokálního model boxu
4. cloud path na pravé straně zmizí
5. zvýrazní se text `runs on your machine`

### On-screen takeaway

- `Cloud = model je jinde`
- `Local = model běží u tebe`

### Manim technika

- `MoveAlongPath`
- `GrowArrow`
- `Circumscribe` na `model weights`

## 6. Scéna 3: Co local dává a nedává

### Cíl

Ukázat tradeoff, ne evangelium.

### Manim objekty

- stylizovaná váha
- vlevo čtyři zelené labely: `control`, `privacy`, `offline`, `long-term cost`
- vpravo čtyři červené labely: `weaker than top cloud`, `setup`, `tuning`, `compromise`

### Animace

1. vytvořit konstrukci váhy
2. postupně přidávat labely na obě strany
3. po každém přidání lehce pootočit rameno váhy
4. na konci vyrovnat do přibližné rovnováhy

### On-screen takeaway

`Výhody ano. Magie ne.`

## 7. Scéna 4: Model -> runtime -> agent system

### Cíl

Dát divákovi základní architekturu.

### Manim objekty

- tři horizontální vrstvy jako stack
- dole `model`
- uprostřed `runtime`
- nahoře `agent system`
- boční příklady jako malé pill labels

### Animace

1. postavit `model`
2. nad něj přirůst `runtime`
3. nad něj přirůst `agent system`
4. z pravé strany postupně přisouvat příklady
5. na chvíli odpojit horní vrstvu a vrátit ji zpět

### On-screen takeaway

- `Model není runtime`
- `Runtime není agent`

### Manim technika

- `GrowFromCenter`
- `ReplacementTransform` mezi abstraktním blokem a popsaným stackem

## 8. Scéna 5: "Utáhne to model"

### Cíl

Rozložit nejasnou větu na měřitelné části.

### Manim objekty

- hlavní claim ve velkém: `Utáhne to 70B model`
- pět checkbox karet:
  - `vejde se`
  - `spustí se`
  - `je rychlý`
  - `udrží context`
  - `drží kvalitu`

### Animace

1. claim se objeví jako velký text
2. claim se zmenší do horní části obrazovky
3. pod něj se rozloží pět karet
4. první dvě dostanou zelenou fajfku
5. zbylé tři se rozsvítí oranžově nebo zůstanou prázdné

### On-screen takeaway

`Vejde se != je použitelné`

## 9. Scéna 6: Quant, active params, context, KV cache

### Cíl

Vysvětlit nejdůležitější technické pojmy co nejčistěji.

### Layout

Dva řádky:

- nahoře `dense vs MoE`
- dole `quant vs KV cache`

### Manim objekty

- dva obdélníky pro dense a MoE
- v MoE jen některé sub-bloky pulzují jako `active`
- slider pro quant
- slider pro context
- rostoucí transparentní cache box

### Animace

1. dense box se celý rozsvítí
2. MoE box se rozpadne na mnoho expert tiles
3. jen malá část expertů se rozsvítí
4. quant slider sníží velikost model boxu
5. context slider zvětší `KV cache`

### On-screen takeaway

- `Total params nejsou celý příběh`
- `Context není zdarma`

### Manim technika

- `ValueTracker`
- updatery pro velikost cache boxu
- `Indicate` na active experts

## 10. Scéna 7: Apple Silicon a unified memory

### Cíl

Ukázat, proč jsou Macy pro local inference zajímavé.

### Manim objekty

- vlevo klasický PC diagram: `CPU`, `RAM`, `GPU`, `VRAM`
- vpravo Apple Silicon diagram: `CPU/GPU/Neural` nad jedním blokem `Unified Memory`

### Animace

1. vlevo se ukáže složitější cesta mezi RAM a VRAM
2. vpravo vznikne jeden velký memory pool
3. velký model tile se zkusí přesunout na obou stranách
4. vlevo se zasekne na úzkém hrdle, vpravo se plynule usadí

### On-screen takeaway

`Pro inference často rozhoduje paměť a bandwidth`

## 11. Scéna 8: Mac mini versus Mac Studio

### Cíl

Zastavit nejčastější faktický zmatek.

### Manim objekty

- dvě vysoké karty vedle sebe
- `Mac mini M4 / M4 Pro`
- `Mac Studio M3 Ultra`
- číselné metric cards

### Metriky na obrazovce

- Mac mini:
  - `120 GB/s`
  - `273 GB/s`
  - `64 GB max unified memory`
- Mac Studio:
  - `819 GB/s`
  - `512 GB unified memory`
  - `Thunderbolt 5`

### Animace

1. obě karty vyrostou
2. čísla se dopisují jedno po druhém
3. mezi kartami se objeví velký červený `!=`
4. přes obraz projede `WarningStamp("NEZAMĚŇOVAT")`

### On-screen takeaway

`Mac mini != Mac Studio`

## 12. Scéna 9: Hardware tier map

### Cíl

Převést hardware na use-case mapu.

### Manim objekty

- tři horizontální lanes
- ikonky use case:
  - `chat`
  - `docs`
  - `coding`
  - `frontier local`
  - `agentic research`

### Animace

1. zobrazit lanes
2. use-case ikony padají shora
3. každá se zastaví na odpovídajícím hardware tieru
4. problematické use cases se odrazí od nižších tierů

### On-screen takeaway

`Ne každý Mac je local AI monster`

## 13. Scéna 10: Local stack pro začátečníka

### Cíl

Ukázat přehled runtime a UI nástrojů.

### Manim objekty

- pět tool cards:
  - `LM Studio`
  - `Ollama`
  - `llama.cpp`
  - `MLX-LM`
  - `Open WebUI`

### Animace

1. toolbox outline
2. karty vyjíždějí jako šuplíky
3. každá dostane jednu krátkou roli
4. na konci se seskupí do doporučeného flow:
   - `LM Studio / Ollama`
   - `MLX-LM / llama.cpp`
   - `Open WebUI`

### On-screen takeaway

`Začni jednoduše`

## 14. Scéna 11: Jak vybírat model

### Cíl

Odvést pozornost od param-count fetishismu.

### Manim objekty

- decision tree
- otázky:
  - `coding?`
  - `multimodal?`
  - `speed?`
  - `RAM budget?`
  - `tool use?`

### Animace

1. kořen stromu: `Jaký model?`
2. větve vyrůstají postupně
3. list nodes se mění na modelové skupiny
4. text `Not just number of B` se objeví pod stromem

### On-screen takeaway

`Neptej se jen na počet B`

## 15. Scéna 12: Speed versus quality

### Cíl

Udělát intuitivní mapu tradeoffů.

### Manim objekty

- scatter plot
- osa X: `tok/s`
- osa Y: `quality`
- bubliny modelových tříd
- toggle `long context`

### Animace

1. nakreslit osy
2. postupně přidat modelové body
3. zapnout `long context`
4. některé body se posunou dolů nebo doprava podle tradeoffu

### On-screen takeaway

`Rychlost, kvalita a context se přetahují`

### Manim technika

- `Axes`
- `Dot`
- `always_redraw` pro body svázané s toggle stavem

## 16. Scéna 13: OpenClaw jako systém

### Cíl

Přerámovat OpenClaw z "aplikace" na orchestrátor.

### Manim objekty

- centrální node `OpenClaw`
- satelity:
  - `dashboard`
  - `channels`
  - `tools`
  - `models`
  - `workspaces`
  - `approvals`

### Animace

1. objeví se střed
2. satelity se rozmístí po kružnici
3. `GrowArrow` od středu ke všem satelitům
4. některé satelity krátce pulsují při zmínce ve voiceoveru

### On-screen takeaway

`OpenClaw = gateway + agent system`

## 17. Scéna 14: OpenClaw onboarding

### Cíl

Ukázat aktuální setup flow.

### Manim objekty

- `TerminalBlock`
- browser-like panel pro dashboard
- malý status badge `Node 22+`

### Terminálové řádky

- `curl -fsSL https://openclaw.ai/install.sh | bash`
- `openclaw onboard --install-daemon`
- `openclaw gateway status`
- `openclaw dashboard`

### Animace

1. terminál se "píše" pomocí `AddTextLetterByLetter`
2. příkazy se odškrtávají
3. transformace terminálu do dashboard panelu
4. v dashboardu se zvýrazní `tasks`, `approvals`, `routing`, `channels`

### On-screen takeaway

- `Není to jen chat v messengeru`
- `Node 22+`

## 18. Scéna 15: OpenClaw a local models

### Cíl

Ukázat, že local model docs jsou střízlivé, ne magické.

### Manim objekty

- dvě cesty:
  - vlevo `small local model`
  - vpravo `large local / hybrid`
- uprostřed gateway

### Animace

1. gateway uprostřed
2. vlevo i vpravo vzniknou dvě pipeline
3. vlevo se objevují červené badge:
   - `weaker planning`
   - `weaker tool use`
   - `higher prompt-injection risk`
4. vpravo zlaté nebo zelené badge:
   - `better planning`
   - `better tool use`
   - `higher cost`

### On-screen takeaway

`Agent chce lepší model než chat`

## 19. Scéna 16: Proč agent potřebuje lepší model

### Cíl

Přidat pocit narůstající komplexity.

### Manim objekty

- workflow chain:
  - `prompt`
  - `plan`
  - `tool`
  - `tool output`
  - `revision`
  - `approval`
  - `final`

### Animace

1. nejdřív jen `prompt -> answer`
2. answer node se transformuje do delší pipeline
3. mezi kroky naskakují malé červené warning triangles

### On-screen takeaway

`Každý krok = další místo, kde se to může rozpadnout`

## 20. Scéna 17: Prompt injection a bezpečnost

### Cíl

Zviditelnit bezpečnost jako součást architektury.

### Manim objekty

- agent node
- web page box
- červený payload text
- gate `approval`
- shield `sandbox`
- workspace boundary

### Animace

1. agent čte web page
2. z web page vyletí červený text `ignore previous instructions`
3. payload míří na agenta
4. mezi ně se vloží `approval gate`
5. objeví se `sandbox` a `workspace isolation`

### On-screen takeaway

`Víc autonomie = víc útokové plochy`

## 21. Scéna 18: Kdy OpenClaw dává smysl

### Cíl

Oddělit chat use case od agent use case.

### Manim objekty

- decision split
- vlevo jednoduchý chat flow
- vpravo multi-step assistant flow

### Animace

1. jeden vstupní node `Co vlastně chceš?`
2. rozdělení na dvě větve
3. vlevo končí v `LM Studio / Ollama / Open WebUI`
4. vpravo pokračuje do `OpenClaw`

### On-screen takeaway

`Nezačínej agentem, pokud chceš jen chat`

## 22. Scéna 19: Doporučené pořadí pro nováčka

### Cíl

Udělát roadmap slide.

### Manim objekty

- horizontální progress line
- sedm milestone dots

### Kroky

1. `run one local model`
2. `learn quant + tok/s`
3. `compare fast vs smart model`
4. `structured outputs + tools`
5. `simple RAG`
6. `OpenAI-compatible endpoint`
7. `agents + OpenClaw`

### Animace

1. linka se kreslí zleva doprava
2. milestone dots se rozsvěcují
3. aktuální milestone dostává `Circumscribe`

### On-screen takeaway

`Nepřeskakuj základy`

## 23. Scéna 20: Cloud vs local vs hybrid

### Cíl

Ukázat hybrid jako realistický kompromis.

### Manim objekty

- tři sloupce:
  - `cloud`
  - `local`
  - `hybrid`
- use-case cards

### Animace

1. tři sloupce vzniknou
2. use-case cards padají shora
3. karty se usazují do nejsmysluplnějšího sloupce
4. `hybrid` postupně nasbírá nejvíc "real world" karet

### On-screen takeaway

`Hybrid bývá nejpraktičtější`

## 24. Scéna 21: Mac versus NVIDIA workstation

### Cíl

Udělát férový tradeoff slide bez flamewar.

### Manim objekty

- dvě dílny nebo dvě velké cards
- `Apple Silicon`
- `NVIDIA workstation`

### Animace

1. dvě cards vedle sebe
2. pod Apple přibývá:
   - `simplicity`
   - `unified memory`
   - `quiet`
3. pod NVIDIA přibývá:
   - `CUDA`
   - `upgrade path`
   - `fine-tuning`
   - `throughput`

### On-screen takeaway

`Mac je appliance, NVIDIA PC je dílna`

## 25. Scéna 22: Největší bullshit

### Cíl

Rychlé energetické rozbití mýtů.

### Manim objekty

- clickbait cards
- red stamps

### Claims

- `už nepotřebuješ cloud`
- `ten quant je skoro jako top API`
- `když běží 70B, mám enterprise`
- `framework vyřeší slabý model`
- `lokální = bezpečné`

### Animace

1. claim cards naskakují rychle za sebou
2. přes každou přijde `WarningStamp("MYTH")` nebo `WarningStamp("DEPENDS")`
3. na konci se všechny claim cards složí do malé hromady a odsunou ze scény

### On-screen takeaway

`Běží != je dobré`

## 26. Scéna 23: Doporučení podle typu člověka

### Cíl

Zakončit video použitelným rozcestníkem.

### Manim objekty

- tři person cards:
  - `nováček`
  - `seriózní local user`
  - `builder osobního asistenta`

### Animace

1. person cards se rozloží do tří sloupců
2. do každé přiběhnou doporučené stack prvky
3. vedle každé se objeví jedna hlavní chyba, které se má vyhnout

### On-screen takeaway

`Jiný cíl = jiný správný setup`

## 27. Scéna 24: Závěr

### Cíl

Vrátit divákovi hlavní mapu celého videa.

### Manim objekty

- tři původní pill labels:
  - `local inference`
  - `self-hosting`
  - `agents`
- pod nimi nové vrstvy:
  - `hardware`
  - `runtime`
  - `security`

### Animace

1. vrátí se původní tři labely z úvodu
2. pod ně se přisune stack dalších vrstev
3. vše se uspořádá do jedné mapy
4. závěrečný claim se napíše doprostřed

### On-screen takeaway

`Když chápeš mapu, hype tě nerozhází`

## 28. Produkční pravidla pro Manim implementaci

### Délka scén

- většina scén `8-18 s`
- myth-busting sekce může být rychlejší `5-8 s`
- onboarding a technical diagrams mohou být delší `15-22 s`

### Layout pravidla

- nenechávat na obrazovce víc než jednu dominantní myšlenku
- při přechodu mezi tématy použít `ReplacementTransform` nebo `FadeTransform`
- nové objekty mají vyrůstat z logiky předchozí scény, ne se jen objevit

### Styl pravidla

- používat geometrii, ne stock ikonky všude
- text držet krátký
- čísla a technické termíny zvýraznit barvou, ne bold spamem
- všechno, co je "riziko", vždy barvit konzistentně červeně
- všechno, co je "správná mentální mapa", držet zeleně

## 29. Doporučená struktura Python souborů

Pokud se to bude implementovat v Manimu, dává smysl rozdělení:

- `manim_project/config.py`
- `manim_project/theme.py`
- `manim_project/components.py`
- `manim_project/scenes_intro.py`
- `manim_project/scenes_local_llm.py`
- `manim_project/scenes_openclaw.py`
- `manim_project/scenes_outro.py`

## 30. Co by šlo udělat jako další krok

Další rozumný krok je z tohoto storyboardu vygenerovat:

1. Manim class list pro každou scénu
2. shared component library
3. skeleton Python soubory s prázdnými `Scene` třídami
4. shot-by-shot voiceover sync markers
