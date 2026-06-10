# 🧗‍♂️ Climbing Vision: From End-to-End to Two-Stage Classification

Projekt badawczo-inżynierski mający na celu automatyczną detekcję interakcji wspinacza ze ścianą (określanie, czy dane kończyny – lewa ręka, prawa ręka, lewa noga, prawa noga – aktywnie trzymają chwyt/stopień) na podstawie nagrań wideo. 

Projekt ewoluował od klasycznego podejścia End-to-End (E2E) do architektury dwuetapowej (Two-Stage Pipeline) wykorzystującej detekcję obiektów (YOLO) i klasyfikację wyciętego obszaru (EfficientNet-B2).

## 🧠 Architektura i Ewolucja Projektu

Podczas prac nad modelem zidentyfikowaliśmy, że głównym wąskim gardłem w analizie wspinaczki za pomocą sieci konwolucyjnych (CNN) jest **rozdzielczość i szum tła**. Gdy na wejście sieci (260x260 px) trafia cała ściana wspinaczkowa, dłoń zajmuje zaledwie kilka pikseli, co uniemożliwia modelowi poprawne wnioskowanie o geometrii chwytu.

Aby rozwiązać ten problem, przetestowaliśmy różne podejścia do reprezentacji danych.

### 🔬 Eksperymenty i Wyniki (BCE Loss)

1. **Baseline E2E (Zbiór Publiczny)**
   * **Opis:** Podawanie pełnych klatek wideo do sieci EfficientNet-B2.
   * **Wynik:** Szybki overfitting (Loss: ~0.43). Sieć uczyła się "na pamięć" koloru tła i specyficznych ścian zamiast mechaniki układu ciała.
2. **Data-Driven E2E (Zbiór Publiczny + Customowy)**
   * **Opis:** Wprowadzenie dużej różnorodności danych (wymieszanie zbioru publicznego z własnymi nagraniami z lokalnych ścianek).
   * **Wynik:** Znaczna poprawa generalizacji (Loss: ~0.39). Problem braku gęstości pikseli (zbyt mała dłoń na obrazku) nadal ograniczał precyzję.
3. **Visual Prompting (Zbiór Customowy + Nakładka YOLO)**
   * **Opis:** Nałożenie kolorowych bounding boxów (wspinacz) i punktów (chwyty) na oryginalne klatki, aby "wskazać" sieci ważne miejsca.
   * **Wynik:** Spadek błędu (Loss: ~0.29). Dowód na to, że sieć radzi sobie lepiej, gdy ułatwi się jej nawigację po kadrze. Metoda mało skalowalna w warunkach komercyjnych (wymagałaby perfekcyjnego detektora wszystkich chwytów).
4. **Ostateczny Pipeline: Two-Stage (Zbiór Publiczny + Customowy + Wycinanie BBox)**
   * **Opis:** Wykorzystanie modelu YOLOv8 do śledzenia wspinacza. Skrypty wycinają sylwetkę wspinacza (z 15% marginesem) i dodają czarny padding w celu zachowania proporcji kwadratu (Aspect Ratio). EfficientNet-B2 klasyfikuje wyłącznie ten wycięty obszar (Region of Interest).
   * **Wynik:** Błąd walidacji **<0.25**. Model radzi sobie bardzo dobrze z częścią kończyn. 

## ⚙️ Pipeline Przetwarzania Danych (Two-Stage)

Zalecany proces przygotowania danych i inferencji składa się z dwóch kroków:

1. **Tracker YOLO (`extract_bboxes.py`):**
   * Przetwarza surowe pliki video.
   * Śledzi wspinacza (algorytm `botsort`) i zapisuje współrzędne sylwetki (x1, y1, x2, y2) klatka po klatce do plików CSV.
2. **Cropper & Padder (`process_data.py` / `process_custom_data.py`):**
   * Czyta pliki wideo oraz wygenerowane pliki BBox CSV.
   * Wycina wspinacza, aplikuje bezpieczny margines (15%) i dodaje czarne paski (Border Padding), aby uniknąć zniekształceń przy operacji Resize do 260x260 px.
   * Kompiluje ostateczny plik `labels.csv` dla wyciętych klatek.

## 🚀 Wymagania i Uruchomienie

### Zależności
```bash
pip install torch torchvision pandas opencv-python tqdm ultralytics