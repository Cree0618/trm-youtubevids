# Weight Formats Video Script And Storyboard

Aktualizovano: 2026-03-28

Tento dokument je produkcni script a storyboard pro Manim video o floating-point a low-precision formatech pouzivanych u LLM inference. Cilem neni jen ukazat nazvy formatu, ale dat divakovi vizualni intuici pro:

- kolik bitu nese samotna hodnota
- jak jsou bity rozdelene mezi sign, exponent a mantissu
- proc se BF16 a FP16 lisi i pri stejne velikosti
- proc FP8, MXFP8 a NVFP4 potrebuji scaling
- jak se zmena precision prepocita do velikosti vah modelu
- proc "4-bit format" neznamena automaticky ctvrtinovou realnou cenu bez overheadu

Hlavni reference jsou v [weight_formats_sources.md](./weight_formats_sources.md).

## 1. Creative Direction

### Cíl videa

Divak ma po zhlednuti chapat tri veci:

1. `Bit layout`
   - FP16, BF16, FP8 a NVFP4 se nelisi jen poctem bitu, ale tim, jak bity rozdeluji mezi range a precision.

2. `Scaling overhead`
   - moderni ultra-low-precision formaty nejsou jen "mensi cislo", ale casto payload plus dodatecne skaly.

3. `Model-size intuition`
   - stejny 7B nebo 70B model muze mit dramaticky jinou pametovou stopu podle formatu.

### Tone

- technicky
- vecny
- bez hype
- kazda scena ma jednu dominantni pointu

### Video length

Priblizne 2.5 az 4 minuty jako samostatny explainer segment nebo vlozka do delsiho videa o kvantizaci a inference.

## 2. Visual Rules

### Background

- tmave technicke pozadi
- ne ciste cerne, ale tmave modro-sede
- zadne dekorativni castice nebo AI klipecky

### Text visibility

Kazdy textovy element musi byt citelny i na mobilu.

Pravidla:

- title: minimum `40-44 px`
- section labels: minimum `24-28 px`
- table/body text: minimum `20-22 px` tam, kde je to realne
- drobne labels ne pod `18 px`

Kazdy svetly text musi mit:

- bud tmavy outline / stroke
- nebo dostatecne tmavy panel za sebou

Doporuceni pro Manim:

- `Text(..., color=WHITE)` davat na tmave filled cards
- pokud text sedi primo na pozadi, pridat tmavy underlay rectangle nebo subtle stroke pres surrounding box
- barevny text nepouzivat bez kontrastniho panelu

### Bit colors

- `sign`: neutral grey
- `exponent`: modra nebo zelena podle format family
- `mantissa`: svetlejsi akcent stejne rodiny
- `scale blocks`: fialova pro MXFP8
- `global scale / NVFP4 metadata`: cervena / oranzova

### Layout discipline

- nepouzivat vic nez 3 hlavni sloupce najednou
- tabulku az na konci
- v kazde scene drzet jasnou hierarchii:
  - hlavni claim
  - hlavni objekt
  - supporting note

## 3. Technical Data To Show

Tyto hodnoty musi byt v animaci nebo doprovodnem textu konzistentni.

### Core bit layouts

- `FP16`
  - 16 bitu
  - `1 sign | 5 exponent | 10 mantissa`

- `BF16`
  - 16 bitu
  - `1 sign | 8 exponent | 7 mantissa`

- `FP8 E4M3`
  - 8 bitu
  - `1 sign | 4 exponent | 3 mantissa`

- `FP8 E5M2`
  - 8 bitu
  - `1 sign | 5 exponent | 2 mantissa`
  - ve videu muze byt jako sekundarni reference vedle E4M3, ne nutne plna samostatna scena

- `MXFP8`
  - FP8 payload
  - block scaling po `32` hodnotach
  - scale ve formatu `E8M0`

- `NVFP4`
  - payload `E2M1`
  - `FP8 E4M3` scale na blok `16` hodnot
  - `FP32` global scale na tensor

### High-level interpretation

- `FP16`
  - vic mantissy
  - mensi exponent range

- `BF16`
  - stejna velikost jako FP16
  - mensi mantissa precision
  - mnohem vetsi dynamic range, prakticky FP32-like exponent range

- `FP8`
  - mensi payload
  - typicky potrebuje scaling strategy

- `MXFP8`
  - neni jen "FP8"
  - je to FP8 plus block-level scaling metadata

- `NVFP4`
  - velmi maly payload
  - funguje diky agresivnejsimu scaling modelu

### Model size examples

Pouzit jednoduche, rychle prepocitatelne priklady bez zbytecneho zahlceni.

Predpoklad:

- `FP16/BF16 = 2 bytes per parameter`
- `FP8 = 1 byte per parameter`
- `FP4 payload = 0.5 byte per parameter`

Na obrazovce vyslovene napsat, ze jde o `raw payload estimate`, a u MXFP8 / NVFP4 dodat, ze skutecna velikost je o neco vyssi kvuli scale metadata.

Doporucene modely:

- `7B model`
  - FP16: `~14 GB`
  - BF16: `~14 GB`
  - FP8: `~7 GB`
  - NVFP4 raw payload: `~3.5 GB`

- `70B model`
  - FP16: `~140 GB`
  - BF16: `~140 GB`
  - FP8: `~70 GB`
  - NVFP4 raw payload: `~35 GB`

- `671B class model`
  - FP16: `~1.34 TB`
  - FP8: `~671 GB`
  - FP4 raw payload: `~335.5 GB`

U posledniho prikladu nepridavat BF16 zvlast, pokud by scena byla preplnena.

## 4. Overall Structure

1. Hook: format neni jen pocet bitu
2. Floating-point anatomy
3. FP16 vs BF16
4. FP8 family
5. MXFP8 block scaling
6. NVFP4 two-level scaling
7. Model-size comparison
8. Final summary table

## 5. Scene-By-Scene Script

## Scene 1: Hook

### Goal

Okamzite zaramovat video jako srovnani formatu, ne jako obecnou kvantizaci.

### Voiceover

`Kdyz se rekne, ze model bezi ve FP16, BF16 nebo treba NVFP4, nejde jen o pocet bitu. Jde o to, kolik z nich padne na range, kolik na precision a kolik dalsi infrastruktury je potreba kolem toho, aby ten format byl vubec pouzitelny.`

### On-screen elements

- Title:
  - `Weight Formats for Inference`
- Subtitle:
  - `Range, precision, scaling, memory footprint`

### Layout

- title centrovane nahore
- subtitle tesne pod nim
- zadne dalsi elementy

### Motion

- jemny fade-in
- subtitle zmizi pri prechodu do dalsi sceny

### Visibility notes

- title na cistem pozadi
- pod title jemny tmavy gradient panel

## Scene 2: Floating-point anatomy

### Goal

Dat divakovi univerzalni mentalni model, na kterem budou stavet dalsi sceny.

### Voiceover

`Vetsina techto formatu je porad nejaka varianta floating-point cisla. To znamena: sign, exponent a mantissa. Exponent urcuje range. Mantissa urcuje jemnost reprezentace. A od toho se pak odviji, jak dobre format snasi vahy, aktivace nebo inferencni skaly.`

### On-screen elements

- Formula:
  - `x = (-1)^s * 2^(e-bias) * (1.m)`
- labels pod formulí:
  - `sign`
  - `exponent = range`
  - `mantissa = precision`

### Layout

- formula ve stredu
- tri male callout arrows nebo brace labels pod ni

### Motion

- formula se napise
- labels priskakuji postupne zleva doprava

### Visibility notes

- formula velka
- labels na tmavych pill badges

## Scene 3: FP16 vs BF16

### Goal

Ukazat, ze stejne 16-bit formaty muzou rozdelovat bity jinak a tim menit chovani.

### Voiceover

`FP16 a BF16 zabiraji uplne stejne misto, ale nejsou zamenitelne. FP16 dava vic bitu do mantissy, a tim padem ma jemnejsi precision. BF16 naopak zachovava osmibitovy exponent jako FP32, takze ma mnohem vetsi dynamic range. To je presne duvod, proc se BF16 tak casto pouziva v modernim trainingu i inferenci.`

### On-screen elements

- dve velke karty vedle sebe
- karta `FP16`
  - `16 bits`
  - `1 | 5 | 10`
  - `more mantissa precision`
  - `smaller exponent range`
- karta `BF16`
  - `16 bits`
  - `1 | 8 | 7`
  - `FP32-like exponent range`
  - `coarser mantissa`

- pod kartami dve barevne bit bars

### Layout

- FP16 vlevo
- BF16 vpravo
- pod nimi horizontální bit bars

### Motion

- nejdriv karty
- potom transformace na bit bars
- nakonec short conclusion caption

### Visibility notes

- cisla bitu uvnitr segmentu musi byt velka
- segmenty nesmi byt moc uzke, radsi sirsi bars

## Scene 4: Add FP8 to the comparison

### Goal

Prepnout z klasickych 16-bit formatu k moderni nizkopresne ceste.

### Voiceover

`Jakmile prejdu na FP8, ziskavam polovicni payload proti 16-bit formatu. Jenze cena za to je mensi exponent a mensi mantissa. Proto samotne FP8 casto nestaci brat jako ciste osmibitovy ekvivalent FP16. Prakticky skoro vzdy prichazi spolu s nejakou scaling strategii.`

### On-screen elements

- treti karta `FP8 E4M3`
  - `8 bits`
  - `1 | 4 | 3`
  - `smaller payload`
  - `typically used with scaling`

- volitelna mala callout note:
  - `E5M2 exists too`
  - `more range, less mantissa`

### Layout

- tri sloupce:
  - FP16
  - BF16
  - FP8 E4M3

### Motion

- kamera zustane stabilni
- FP8 karta se vysune zprava
- bit bar FP8 se zarovna pod ostatni

### Visibility notes

- nepokracovat s plnou tabulkou v teto scene
- focus na jeden claim:
  - `8 bits are not enough context by themselves`

## Scene 5: MXFP8 block scaling

### Goal

Vysvetlit, ze nektere formaty jsou payload plus metadata, ne jen payload samotny.

### Voiceover

`MXFP8 uz neni jen samotny FP8 payload. Je to FP8 hodnota doplnena o block-level scaling. Typicky jedna skala na 32 hodnot. Tohle zmensuje lokalni chybu a dela nizkou precision podstatne pouzitelnejsi. Ale zaroven uz nejde o cisty osmibitovy svet bez overheadu.`

### On-screen elements

- karta `MXFP8`
  - `FP8 payload`
  - `block scale / 32 values`
  - `E8M0 scale`

- 32 malych ctvercu ve dvou radach
- nad nimi napis `32 values`
- pod nimi `1 shared scale`

### Layout

- vlevo karta
- vpravo 32-value block diagram

### Motion

- block se vyplni po skupinach
- jeden scale token "zaklapne" na cely blok

### Visibility notes

- ctverce musi byt dost velke, aby bylo jasne, ze jde o skupinu
- labels spise mimo ctverce nez uvnitr

## Scene 6: NVFP4 two-level scaling

### Goal

Ukazat, ze NVFP4 neni "proste 4 bity", ale 4-bit payload s viceurovnovym scalingem.

### Voiceover

`NVFP4 jde jeste dal. Samotny payload je jen ctyrbitovy floating-point format E2M1. Aby ale byl tak agresivne maly format pouzitelny, pridava se FP8 block scale typicky na 16 hodnot a jeste jedna globalni FP32 skala na cely tensor. Takze kdyz nekdo rekne FP4, je dulezite se ptat: mysli tim opravdu jen payload, nebo celou reprezentaci vcetne scaling metadata?`

### On-screen elements

- karta `NVFP4`
  - `E2M1 payload`
  - `FP8 scale / 16`
  - `FP32 tensor scale`

- matematicka formule:
  - `x = x_E2M1 * s_block * s_global`

- diagram:
  - 16 ctvercu jako jeden blok
  - jedna block scale badge
  - jedna global scale badge pro cely tensor

### Layout

- karta vlevo
- diagram vpravo
- formule dole pod diagramem

### Motion

- nejdriv payload squares
- potom block scale
- potom global scale
- nakonec formule

### Visibility notes

- formule musi byt velka a kratka
- labels `block scale` a `global scale` nenechat prekryt

## Scene 7: Size comparison with model examples

### Goal

Prevest bity do intuice o velikosti realnych modelu.

### Voiceover

`Ted ta prakticka cast. Kdyz vezmu 7B model, raw payload ve FP16 nebo BF16 je zhruba 14 gigabajtu. Ve FP8 zhruba 7 gigabajtu. A u FP4 payloadu kolem 3.5 gigabajtu, samozrejme s tim, ze realna reprezentace muze byt o neco vetsi kvuli scaling metadata. U 70B modelu se ten rozdil nasobi deseti. A presne tady je videt, proc low-precision formaty dramaticky meni to, co se vejde do jedne GPU nebo do lokalniho stroje.`

### On-screen elements

Pouzit tri model examples jako tri radky:

- `7B model`
- `70B model`
- `671B class model`

Kazdy radek ma 4 horizontalni bars:

- FP16/BF16
- FP8
- NVFP4 raw payload
- optional grey note `+ scaling overhead`

Konretni text:

- `7B`
  - FP16/BF16 `~14 GB`
  - FP8 `~7 GB`
  - NVFP4 raw `~3.5 GB`

- `70B`
  - FP16/BF16 `~140 GB`
  - FP8 `~70 GB`
  - NVFP4 raw `~35 GB`

- `671B`
  - FP16 `~1.34 TB`
  - FP8 `~671 GB`
  - FP4 raw `~335.5 GB`

### Layout

- vlevo labels model sizes
- vpravo bars
- nejdelsi bar je FP16 reference
- vse ostatni je relativne skalovane k ni

### Motion

- nejdriv objevit `7B`
- pak morph do `70B`
- pak pridat `671B class`
- na zaver vsechny tri najednou jako stacked chart

### Visibility notes

- ciselne velikosti nepsat dovnitr kratkych baru, pokud se nevejdou
- v takovem pripade dat label napravo od baru
- explicitne napsat:
  - `raw payload estimate`
  - `real deployment size depends on metadata, layout, kernels, cache`

## Scene 8: Final comparison table

### Goal

Uzavrit video do jedne tabulky, kterou si divak snadno zapamatuje.

### Voiceover

`Takze shrnuti: FP16 a BF16 maji stejny rozmer, ale jiny kompromis mezi range a precision. FP8 zmensuje payload, ale casto potrebuje scaling strategii. MXFP8 a NVFP4 uz nejsou jen samotne payload formaty, ale cele reprezentace se sdilenymi skalami. A to je duvod, proc nestaci ptat se jen na pocet bitu. Je potreba se ptat, jak ten format vypada jako celek.`

### On-screen elements

Tabulka:

- `Format`
- `Payload bits`
- `Range / precision profile`
- `Scaling requirement`
- `Raw weight footprint`

Rows:

- FP16
- BF16
- FP8 E4M3
- MXFP8
- NVFP4

### Layout

- cela tabulka vycentrovana
- pod ni jedno concluding line:
  - `More bits buy precision. Better scaling buys usable low precision.`

### Visibility notes

- tabulka max 5 radku
- zadny text mensi nez 18 px
- pokud bude tabulka preplnena, zkratit wording, ne zmensovat font

## 6. Implementation Notes For Manim

### Reusable components

- `FormatCard`
- `BitBar`
- `BlockScaleDiagram`
- `ModelSizeBarRow`
- `SummaryTable`

### Text handling

- vsechny Text objekty bud:
  - na tmavem filled panelu
  - nebo s jemnym kontrastnim podkladem

### Safe margins

- nenechat text bliz nez `0.35` az `0.45` jednotky od kraje frame
- tabulky vzdy radsi zmensit az po kontrole screenshotu

### Rendering checks

Pred final renderem udelat:

1. screenshot kazde sceny
2. kontrolu mobilni citelnosti
3. kontrolu, ze zadny label neleze pres bar nebo formula box
4. kontrolu, ze bit blocks nejsou moc male

## 7. Data Safety Notes

Vsechny velikosti modelu ve videu oznacit jako:

- `rough raw weight estimate`

Nevydavat je za presnou deployovanou velikost, protoze realna velikost zalezi i na:

- metadata overhead
- serialization format
- alignment
- scales
- packing
- pripadne dalich engine-specific strukturach

## 8. Deliverable Intent

Finalni video musi divakovi zanechat tuto vetu:

`Float format neni jen pocet bitu. Je to kompromis mezi rangem, precision a overheadem, ktery primo urcuje, jak velky model se vejde do pameti a jak spolehlive pujde provozovat v nizke presnosti.`
