# Sprawozdanie - BetaZone AI heuristic

## Cel projektu

Celem projektu było sprawdzenie, czy model wizyjny jest w stanie automatycznie rozpoznawać, które kończyny wspinacza aktywnie korzystają z chwytów lub stopni na ściance wspinaczkowej. Problem został sformułowany jako klasyfikacja wieloetykietowa dla czterech klas:

- `LH` - lewa ręka,
- `RH` - prawa ręka,
- `LF` - lewa noga,
- `RF` - prawa noga.

Dla każdej klatki model zwraca cztery niezależne prawdopodobieństwa. Wartość powyżej progu `0.5` interpretowano jako aktywny kontakt danej kończyny z chwytem lub stopniem.

We wszystkich wariantach końcowym klasyfikatorem był EfficientNet-B2. Model pracował na wejściu `260x260`, miał cztery wyjścia z aktywacją `sigmoid`, był uczony funkcją straty `BCELoss`, a w klasyfikatorze zastosowano `Dropout(p=0.6)`, aby ograniczyć przeuczenie.

## Dataset i przygotowanie danych

W projekcie wykorzystano dwa źródła danych. Pierwszym był zewnętrzny dataset `the-way-up`, zawierający nagrania wspinaczy na różnych drogach oraz pliki z informacją, w których przedziałach klatek dana kończyna korzystała z chwytu albo stopnia. Drugim źródłem były własne nagrania wykonane na potrzeby projektu. Były one szczególnie ważne, ponieważ celem było uzyskanie działania sensownego właśnie na naszych filmach, a nie tylko na materiale z gotowego datasetu.

Z obu źródeł dane sprowadzono do wspólnego formatu klasyfikacji wieloetykietowej. Dla każdej wybranej klatki tworzony był wektor czterech wartości binarnych: `LH`, `RH`, `LF`, `RF`. Wartość `1` oznaczała aktywny kontakt kończyny z chwytem lub stopniem, a `0` brak takiego kontaktu. W przypadku datasetu `the-way-up` etykiety były odczytywane z przedziałów `startFrame` - `endFrame` w plikach `holdUsage`. Dla własnych nagrań korzystano z przygotowanych plików CSV z etykietami per klatka.

Własne nagrania miały dodatkowe znaczenie dla dwóch pipeline'ów z preprocessingiem. Dla nich przygotowano bounding boxy wspinacza oraz wersje filmów z naniesionymi maskami chwytów. Wariant E2E i wariant z cropem mogły korzystać zarówno z `the-way-up`, jak i z naszych nagrań. Wariant z maskami chwytów był praktycznie ograniczony do własnego datasetu, ponieważ maski YOLO i ich interpolacja działały stabilnie głównie na naszych filmach. Wariant z wycinaniem wspinacza był bardziej uniwersalny, bo wymagał przede wszystkim bboxa osoby, a nie dokładnej maski każdego chwytu.

## Metodologia eksperymentu

Eksperyment składał się z kilku etapów. Najpierw z nagrań wyodrębniano klatki i przypisywano im etykiety czterech kończyn. Następnie dla każdej klatki przygotowywano obraz wejściowy zgodnie z jednym z trzech wariantów: pełna klatka, klatka z maskami i bboxem albo crop wokół wspinacza. Tak przygotowane dane trafiały do EfficientNet-B2, który zwracał cztery niezależne prawdopodobieństwa kontaktu.

Podział danych wykonywano na poziomie filmów, a nie pojedynczych klatek. Jest to istotne, ponieważ sąsiednie klatki z tego samego nagrania są do siebie bardzo podobne. Losowe mieszanie klatek między treningiem i testem mogłoby sztucznie zawyżyć wyniki, bo model widziałby prawie te same sytuacje w obu zbiorach. W finalnym porównaniu opisanym niżej wszystkie trzy modele oceniono na tych samych dwóch filmach testowych, żeby wyniki były bezpośrednio porównywalne.

Trening każdego wariantu polegał na uczeniu tego samego klasyfikatora, ale na innym typie obrazu wejściowego. Dzięki temu porównanie pokazywało głównie wpływ preprocessingu, a nie różnic w architekturze modelu. Do ewaluacji użyto progu `0.5` dla każdego z czterech wyjść. Oprócz ogólnych metryk liczono też wyniki osobno dla każdej kończyny i osobno dla każdego filmu, bo w tym zadaniu pojedyncza wartość accuracy może ukrywać problemy na konkretnej klasie, np. dla jednej nogi.

## Porównywane podejścia

### 1. Model E2E na pełnych klatkach

Pierwszy wariant był najprostszym podejściem end-to-end. Na wejście EfficientNet-B2 trafiała pełna, surowa klatka wideo. Model nie dostawał jawnej informacji o położeniu chwytów ani o pozycji wspinacza.

Zaletą tego podejścia jest prostota: nie wymaga ono dodatkowych modeli detekcji ani preprocessingu. Wadą jest bardzo duża ilość szumu w obrazie. Model musi sam nauczyć się znaleźć wspinacza, odróżnić kończyny od tła i rozpoznać kontakt z chwytem. W praktyce okazało się to najtrudniejszym wariantem.

### 2. Model z maskami chwytów i bboxem wspinacza

Drugi wariant polegał na wzbogaceniu obrazu wejściowego o informacje semantyczne. Na klatki nakładano maski chwytów oraz bounding box wspinacza. Maski chwytów były wyznaczane modelem YOLO, a pozycja wspinacza była określana przez detekcję osoby.

Ponieważ detekcja chwytów na pojedynczych klatkach była niestabilna, zastosowano interpolację masek w czasie. W praktyce chodziło o uzupełnianie i stabilizowanie informacji o chwytach między kolejnymi klatkami, tak aby maski nie znikały nagle przy chwilowej okluzji albo słabszej detekcji. Dzięki temu model nie musiał samodzielnie szukać chwytów w obrazie, tylko dostawał je jako wizualną podpowiedź.

Ograniczeniem tego podejścia jest zależność od jakości masek. Pipeline działał sensownie głównie na własnych nagraniach, ponieważ interpolacja masek dla materiałów z zewnętrznego datasetu była zbyt zawodna.

### 3. Model z wyciętym wspinaczem

Trzeci wariant używał bounding boxa wspinacza do wycięcia z klatki tylko obszaru zajmowanego przez osobę. Crop był rozszerzany o margines, dopełniany do kwadratu i skalowany do wejścia EfficientNet-B2.

W tym podejściu model nadal nie dostaje jawnych masek chwytów, ale analizuje znacznie mniejszy i bardziej istotny fragment obrazu. Redukuje to wpływ tła, pustych fragmentów ściany i przypadkowych obiektów w kadrze. Przy brakujących detekcjach bboxa można używać sąsiednich ramek lub interpolowanych współrzędnych, żeby utrzymać ciągłość wejścia dla modelu.

## Zbiór testowy

Finalna ewaluacja została wykonana na tych samych dwóch nagraniach dla wszystkich trzech modeli: `IMG_0903` oraz `IMG_0899`. Dzięki temu wyniki są bezpośrednio porównywalne. Łącznie oceniono `2116` oznaczonych klatek.

| Film | Liczba klatek | Pokrycie bbox | LH+ | RH+ | LF+ | RF+ |
|---|---:|---:|---:|---:|---:|---:|
| IMG_0903 | 699 | 91.1% | 76.3% | 72.1% | 79.5% | 71.2% |
| IMG_0899 | 1417 | 95.3% | 85.7% | 82.6% | 87.2% | 76.1% |

Warto zauważyć, że dane są niezbalansowane: dla większości klatek kontakt kończyny z chwytem lub stopniem jest oznaczony jako aktywny. Dlatego oprócz `Macro F1` analizowano także `exact match accuracy` i `Hamming loss`.

## Metryki

Do oceny wykorzystano:

- `BCE loss` - średni błąd probabilistyczny modelu,
- `exact match accuracy` - odsetek klatek, w których model poprawnie przewidział cały wektor czterech etykiet,
- `Hamming loss` - odsetek błędnych decyzji liczony osobno dla każdej kończyny,
- `Macro F1` - średnia F1 po czterech kończynach,
- `Micro F1` - F1 liczona globalnie po wszystkich decyzjach,
- `FPS` - liczba klatek przetwarzanych na sekundę w trakcie ewaluacji.

## Wyniki globalne

| Model | Wejście | BCE loss | Exact match | Hamming loss | Macro F1 | Micro F1 | FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| E2E | pełna klatka | 0.884 | 0.025 | 0.426 | 0.645 | 0.704 | 41.5 |
| Maski + bbox | klatka z maskami i bboxem | 0.311 | 0.628 | 0.106 | 0.935 | 0.936 | 92.4 |
| BBox crop | wycięty wspinacz | 0.301 | 0.648 | 0.096 | 0.943 | 0.943 | 91.8 |

Najlepszy wynik globalny uzyskał wariant `BBox crop`. Miał najniższy `BCE loss`, najwyższy `exact match accuracy` i najniższy `Hamming loss`. Wariant z maskami chwytów był bardzo blisko, ale minimalnie przegrał w uśrednieniu po obu filmach.

Czysty model E2E wypadł zdecydowanie najsłabiej. `Exact match` na poziomie `0.025` oznacza, że model prawie nigdy nie trafiał jednocześnie wszystkich czterech etykiet dla jednej klatki. To sugeruje, że sama pełna klatka nie dostarczała modelowi informacji w formie wystarczająco łatwej do wykorzystania.

## Wyniki per kończyna

| Model | LH F1 | RH F1 | LF F1 | RF F1 |
|---|---:|---:|---:|---:|
| E2E | 0.868 | 0.846 | 0.206 | 0.662 |
| Maski + bbox | 0.948 | 0.928 | 0.932 | 0.935 |
| BBox crop | 0.950 | 0.934 | 0.971 | 0.915 |

`BBox crop` najlepiej rozpoznawał lewą rękę, prawą rękę i lewą nogę. Wariant z maskami był najlepszy dla prawej nogi. Różnice między tymi dwoma wariantami są niewielkie, więc oba można uznać za skuteczne.

Największy problem modelu E2E dotyczył lewej nogi. Dla klasy `LF` uzyskał F1 równe tylko `0.206`, przy `1582` błędach false negative. Oznacza to, że model bardzo często nie wykrywał aktywnego kontaktu lewej nogi, mimo że był on obecny w etykietach.

## Wyniki per film

| Film | E2E exact / Macro F1 | Maski + bbox exact / Macro F1 | BBox crop exact / Macro F1 |
|---|---:|---:|---:|
| IMG_0899 | 0.018 / 0.627 | 0.622 / 0.935 | 0.664 / 0.947 |
| IMG_0903 | 0.037 / 0.598 | 0.639 / 0.937 | 0.615 / 0.934 |

Na `IMG_0899` najlepszy był `BBox crop`. Na `IMG_0903` minimalnie lepszy był wariant `Maski + bbox`. To pokazuje, że oba warianty z preprocessingiem są porównywalne jakościowo, ale crop był nieco stabilniejszy w wyniku końcowym.

## Porównanie z baseline

Dla tego zbioru testowego prosty baseline "zawsze przewiduj kontakt" osiągał:

| Baseline | Exact match | Hamming loss | Macro F1 |
|---|---:|---:|---:|
| Zawsze kontakt | 0.538 | 0.198 | 0.890 |

Oba warianty z preprocessingiem wyraźnie poprawiają `Hamming loss`: model z maskami do `0.106`, a `BBox crop` do `0.096`. Oznacza to, że modele nie tylko korzystają z niezbalansowania danych, ale realnie ograniczają liczbę błędnych decyzji per kończyna.

E2E jest natomiast gorszy od baseline pod względem `exact match`, `Hamming loss` i `Macro F1`. W tej konfiguracji nie można uznać go za skuteczne rozwiązanie.

## Wnioski

Najważniejszy wniosek jest taki, że EfficientNet-B2 potrzebuje zawężenia informacji wejściowej. Pełna klatka zawiera za dużo nieistotnego kontekstu, a model ma problem z samodzielnym odnalezieniem człowieka, kończyn i relacji kończyn z chwytami.

Najlepszym wariantem w obecnym eksperymencie okazał się `BBox crop`. Jest prostszy niż pipeline z maskami, działa szybciej niż E2E i osiąga najlepsze metryki globalne. Wariant z maskami chwytów również działa dobrze i miejscami wygrywa z cropem, ale wymaga bardziej złożonego preprocessingu oraz stabilnej interpolacji masek.

W praktyce najbardziej obiecujące są więc podejścia dwuetapowe: najpierw wykrycie istotnych elementów sceny, a dopiero potem klasyfikacja kontaktu kończyn.

## Możliwe kierunki dalszego rozwoju

Pierwszym kierunkiem byłoby dodanie estymacji pozy człowieka, np. przez MediaPipe. Wyniki E2E pokazują, że EfficientNet miał wyraźny problem z samodzielnym rozpoznaniem sylwetki i kończyn w pełnej klatce. Punkty szkieletu mogłyby dać modelowi jawną informację o położeniu dłoni i stóp.

To podejście nie jest jednak trywialne. We wspinaczce często występują okluzje: ręce zasłaniają chwyty, nogi są częściowo poza kadrem, a ciało przyjmuje nietypowe pozycje. W takich przypadkach interpolacja punktów szkieletu byłaby skomplikowana i podatna na błędy.

Drugim, prawdopodobnie najprostszym praktycznie krokiem, byłoby połączenie masek YOLO z wersją cropped. Model dostawałby jednocześnie wycięty obszar wokół wspinacza oraz informację o położeniu chwytów. Taki wariant mógłby połączyć zalety `BBox crop` i podejścia `Maski + bbox`.

Ograniczeniem jest to, że interpolacja masek YOLO działa obecnie tylko na naszym własnym datasecie. Aby to podejście było wiarygodne, potrzebny byłby znacznie większy własny zbiór nagrań: różne drogi, różne kolory chwytów, różne kąty kamery, inne oświetlenie i więcej stylów wspinania.

## Podsumowanie

Projekt pokazał, że klasyfikacja kontaktu kończyn ze ścianą jest możliwa, ale wymaga odpowiedniego przygotowania danych wejściowych. Najlepszy wynik osiągnięto po ograniczeniu obrazu do obszaru wspinacza. Sam EfficientNet-B2 na pełnej klatce nie poradził sobie z zadaniem w sposób stabilny.

Aktualnie najlepszym kompromisem między skutecznością i prostotą jest `BBox crop`. Wariant z maskami chwytów pozostaje bardzo obiecujący, ale wymaga większego datasetu i stabilniejszego pipeline'u interpolacji.
