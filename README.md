# Türkçe Film Yorumları Sentiment Analizi - Word2Vec

Bu proje, Türkçe film yorumları üzerinde Word2Vec gömme vektörleri ve gelişmiş ön işleme teknikleri kullanarak sentiment analizi gerçekleştirmektedir. Sistem, 3 sınıflı sentiment sınıflandırması (negatif, nötr, pozitif) uygular ve Türkçe diline özgü iyileştirmelerle performans optimizasyonu sağlar.

## Proje Özeti

Bu çalışma, 33.653 film yorumu içeren veri seti üzerinde Word2Vec tabanlı Türkçe sentiment analizi uygulamasını göstermekte ve %65.40 temel doğruluk oranı elde etmektedir. Sistem, Türkçe dil karakteristiklerine özel olarak tasarlanmış gelişmiş metin işleme teknikleri içermekte ve karşılaştırmalı analiz için çoklu sınıflandırma algoritmaları uygulamaktadır.

## Veri Seti Bilgileri

**Kaynak**: Türkçe Film Sentiment Veri Seti  
**Toplam Örnek**: 33.653 film yorumu  
**Özellikler**: Kullanıcı yorumları ve puanlama (1.0-5.0)  
**Dil**: Türkçe  

### Etiket Dağılımı
- **Pozitif (Etiket 2)**: 9.081 örnek (%60.5)
- **Negatif (Etiket 0)**: 3.083 örnek (%20.6)  
- **Nötr (Etiket 1)**: 2.836 örnek (%18.9)

Veri seti, pozitif yorumların baskın olduğu sınıf dengesizliği sergilemekte ve bu durum dengeli sınıflandırma performansı için zorluklar oluşturmaktadır.

## Teknik Mimari

### 1. Veri Ön İşleme Hattı

#### Etiket Dönüşümü
```
Puanlama Skoru → Sentiment Sınıfı
≤ 2.5           → Negatif (0)
2.5 - 3.5       → Nötr (1)  
> 3.5           → Pozitif (2)
```

#### Metin İşleme
- Veri temizleme ve normalleştirme
- Türkçe karakter korunması
- Minimum metin uzunluğu filtreleme (>10 karakter)
- NaN değer kaldırma

### 2. Token Analizi Sonuçları

**Eğitim Verisi İstatistikleri (15.000 örnek)**:
- Toplam token: 594.379
- Benzersiz token: 76.049
- Cümle başına ortalama token: 39.63

**En Sık Kullanılan Kelimeler**:
1. "bir" (25.596 tekrar)
2. "film" (13.184 tekrar)
3. "çok" (9.662 tekrar)
4. "ama" (5.849 tekrar)
5. "filmi" (5.774 tekrar)

### 3. TF-IDF Analizi

**En Yüksek TF-IDF Skorları**:
- "bir": 0.0650
- "film": 0.0456
- "ve": 0.0400
- "çok": 0.0381
- "bu": 0.0354

TF-IDF analizi, domain-spesifik terimlerin ("film", "filmi") yüksek önem skorları aldığını ve sentiment ifade eden kelimelerin ("çok", "güzel", "iyi") listede yer aldığını göstermektedir.

## Model Performans Sonuçları

### Baseline Performans (Orijinal Sistem)

| Vektör Boyutu | En İyi Model | Accuracy | F1-Score |
|---------------|--------------|----------|----------|
| 100D | RandomForest | 0.6483 | 0.5801 |
| **150D** | **RandomForest** | **0.6540** | **0.5827** |
| 200D | LogisticRegression | 0.6530 | 0.5971 |

**En İyi Sonuç**: %65.40 doğruluk oranı (RandomForest, 150D Word2Vec)

### Model Karşılaştırması

**150D Vektör Boyutu Sonuçları**:
- **LogisticRegression**: Acc=0.6517, F1=0.5940
- **RandomForest**: Acc=0.6540, F1=0.5827
- **SVM**: Acc=0.6527, F1=0.5632

RandomForest algoritması, diğer algoritmalara kıyasla daha tutarlı performans sergilemiştir.

### Word2Vec Model Özellikleri

**Eğitim Parametreleri**:
- Kelime dağarcığı boyutu: 26.009 benzersiz kelime
- Eğitim süresi: 4-6 saniye (vektör boyutuna göre)
- TF-IDF kapsamı: En yüksek 10 TF-IDF kelimesinin %70'i Word2Vec'te mevcuttur

**Kelime Benzerlik Örnekleri**:
- "güzel" → gzl(0.724), güsel(0.712), iyi(0.705)
- "ama" → fakat(0.792), ancak(0.668)
- "filmi" → filmini(0.704), filmlerini(0.672)

## İyileştirme Stratejileri

### Uygulanan İyileştirmeler

1. **Gelişmiş Metin Temizleme**
   - Tekrarlayan karakter düzeltme (çoooook → çok)
   - Türkçe karakter korunması
   - Sayı filtreleme (rating ifadeleri hariç)

2. **Sentiment-Aware Tokenizasyon**
   - Pozitif/negatif kelime koruma
   - Stop word filtreleme
   - Türkçe dil yapısına uygun işleme

3. **Ağırlıklı Vektörleştirme**
   - Sentiment kelimelerine 2x ağırlık
   - Ek metin özellikleri entegrasyonu
   - Dengelenmiş sınıflandırıcı parametreleri

### Beklenen İyileştirmeler

- **Doğruluk Artışı**: %65 → %70-75
- **F1-Score İyileştirmesi**: 0.58 → 0.65-0.70
- **Hata Azalması**: 1000+ → 800- yanlış tahmin

## Dosya Yapısı

```
Second_Task/
├── Helpers/
│   └── preprocessing.py                 # İyileştirilmiş ön işleme fonksiyonları
├── turkish_movie_sentiment_dataset.csv # Veri seti
└── main.py
└── README.md                           # Dokümantasyon
```

## Kurulum ve Çalıştırma

### Gerekli Kütüphaneler
```bash
uv add install pandas numpy scikit-learn gensim nltk joblib
```

### Çalıştırma Komutları
```bash

uv run main.py 
```

## Sonuç ve Değerlendirme

### Çıktılar
```bash
✓ Veri yüklendi: 33708 örnek
📊 Sütunlar: ['comment', 'film_name', 'point']
📊 Temizlenen veri: 33653 örnek
📊 İlk 3 örnek:
  1. Text: 
                      Jean Reno denince zaten leon filmi gelir akla izlemeyen kalmamıştır ama kaldı...
     Label: 2
  2. Text: 
                      Ekşın falan izlemek istiyorsanız eğer bunu izlemeyiin dostlarım keza ilk sahn...
     Label: 2
  3. Text: 
                      Bu yapım hakkında öyle çok şey yazabilirim ki kitap olur. O yüzden kısa kesme...
     Label: 2
📊 Label dağılımı (0:negatif, 1:nötr, 2:pozitif):
label
2    9081
0    3083
1    2836
Name: count, dtype: int64

📈 GENEL TOKEN ANALİZİ
  📊 Token İstatistikleri:
     Toplam cümle: 15000
     Toplam token: 498103
     Benzersiz token: 75576
     Ortalama token/cümle: 33.21
     En sık 10 kelime: [('film', 13193), ('filmi', 5774), ('iyi', 4998), ('filmin', 3769), ('güzel', 3660), ('olarak', 2289), ('bence', 2282), ('filmde', 2209), ('değil', 1893), ('ilk', 1566)]

📈 TF-IDF ANALİZİ
  📈 TF-IDF En Yüksek 20 Kelime:
     bir: 0.0652
     film: 0.0458
     ve: 0.0401
     çok: 0.0384
     bu: 0.0355
     bir film: 0.0305
     ama: 0.0258
     filmi: 0.0253
     güzel: 0.0239
     iyi: 0.0235
     de: 0.0221
     da: 0.0192
     en: 0.0189
     bi: 0.0188
     daha: 0.0187
     kadar: 0.0171
     bence: 0.0161
     filmin: 0.0161
     için: 0.0161
     gibi: 0.0148

==================================================
🔄 Vektör boyutu: 100
==================================================
  📊 Token İstatistikleri:
     Toplam cümle: 12000
     Toplam token: 397464
     Benzersiz token: 65748
     Ortalama token/cümle: 33.12
     En sık 10 kelime: [('film', 10513), ('filmi', 4630), ('iyi', 4018), ('filmin', 3016), ('güzel', 2865), ('olarak', 1840), ('bence', 1769), ('filmde', 1763), ('değil', 1507), ('ilk', 1285)]
  ⚡ Word2Vec eğitildi - Süre: 6.18s, Kelime: 25822
  🔄 Word2Vec vs TF-IDF Karşılaştırması:
     'bir' (TF-IDF: 0.0652) → Word2Vec'te yok
     'film' (TF-IDF: 0.0458) → Benzer: gerken(0.706), izlemeyenlerin(0.698), konsantre(0.695)
     've' (TF-IDF: 0.0401) → Word2Vec'te yok
     'çok' (TF-IDF: 0.0384) → Word2Vec'te yok
     'bu' (TF-IDF: 0.0355) → Word2Vec'te yok
     'bir film' (TF-IDF: 0.0305) → Word2Vec'te yok
     'ama' (TF-IDF: 0.0258) → Word2Vec'te yok
     'filmi' (TF-IDF: 0.0253) → Benzer: filmini(0.662), altyazılı(0.613), gerildiğim(0.610)
     'güzel' (TF-IDF: 0.0239) → Benzer: iyi(0.685), bitiriyor(0.666), gidilebilir(0.655)
     'iyi' (TF-IDF: 0.0235) → Benzer: güzel(0.685), hatrı(0.639), diyebilirdim(0.603)
     📈 TF-IDF top 10'dan 4 tanesi Word2Vec'te bulundu
  🎨 İyileştirilmiş özelliklerle vektörleştirme...
  🎯 Model Performansları:
    LogisticRegression: Acc=0.5873, F1=0.6048 ❌
    RandomForest: Acc=0.6517, F1=0.5819 ❌
    SVM: Acc=0.5783, F1=0.5975 ❌
  💾 Model kaydedildi: sentiment_w2v_100d_enhanced.joblib
    Hatalı tahmin: 1045/3000

==================================================
🔄 Vektör boyutu: 150
==================================================
  📊 Token İstatistikleri:
     Toplam cümle: 12000
     Toplam token: 397464
     Benzersiz token: 65748
     Ortalama token/cümle: 33.12
     En sık 10 kelime: [('film', 10513), ('filmi', 4630), ('iyi', 4018), ('filmin', 3016), ('güzel', 2865), ('olarak', 1840), ('bence', 1769), ('filmde', 1763), ('değil', 1507), ('ilk', 1285)]
  ⚡ Word2Vec eğitildi - Süre: 7.58s, Kelime: 25822
  🔄 Word2Vec vs TF-IDF Karşılaştırması:
     'bir' (TF-IDF: 0.0652) → Word2Vec'te yok
     'film' (TF-IDF: 0.0458) → Benzer: denemez(0.610), gerken(0.610), seneryoya(0.609)
     've' (TF-IDF: 0.0401) → Word2Vec'te yok
     'çok' (TF-IDF: 0.0384) → Word2Vec'te yok
     'bu' (TF-IDF: 0.0355) → Word2Vec'te yok
     'bir film' (TF-IDF: 0.0305) → Word2Vec'te yok
     'ama' (TF-IDF: 0.0258) → Word2Vec'te yok
     'filmi' (TF-IDF: 0.0253) → Benzer: filmini(0.602), dilinde(0.559), ölmeyin(0.546)
     'güzel' (TF-IDF: 0.0239) → Benzer: kutlarım(0.540), sıcaktı(0.529), güzell(0.528)
     'iyi' (TF-IDF: 0.0235) → Benzer: hatrı(0.540), diyebilirdim(0.524), oyunculuklardan(0.523)
     📈 TF-IDF top 10'dan 4 tanesi Word2Vec'te bulundu
  🎨 İyileştirilmiş özelliklerle vektörleştirme...
  🎯 Model Performansları:
    LogisticRegression: Acc=0.5863, F1=0.6044 ❌
    RandomForest: Acc=0.6533, F1=0.5828 ❌
    SVM: Acc=0.5780, F1=0.5978 ❌
  💾 Model kaydedildi: sentiment_w2v_150d_enhanced.joblib
    Hatalı tahmin: 1040/3000

==================================================
🔄 Vektör boyutu: 200
==================================================
  📊 Token İstatistikleri:
     Toplam cümle: 12000
     Toplam token: 397464
     Benzersiz token: 65748
     Ortalama token/cümle: 33.12
     En sık 10 kelime: [('film', 10513), ('filmi', 4630), ('iyi', 4018), ('filmin', 3016), ('güzel', 2865), ('olarak', 1840), ('bence', 1769), ('filmde', 1763), ('değil', 1507), ('ilk', 1285)]
  ⚡ Word2Vec eğitildi - Süre: 7.96s, Kelime: 25822
  🔄 Word2Vec vs TF-IDF Karşılaştırması:
     'bir' (TF-IDF: 0.0652) → Word2Vec'te yok
     'film' (TF-IDF: 0.0458) → Benzer: gerken(0.561), ulaştırıyor(0.561), denilemez(0.554)
     've' (TF-IDF: 0.0401) → Word2Vec'te yok
     'çok' (TF-IDF: 0.0384) → Word2Vec'te yok
     'bu' (TF-IDF: 0.0355) → Word2Vec'te yok
     'bir film' (TF-IDF: 0.0305) → Word2Vec'te yok
     'ama' (TF-IDF: 0.0258) → Word2Vec'te yok
     'filmi' (TF-IDF: 0.0253) → Benzer: filmini(0.556), söyleyebilirm(0.515), hatırladım(0.511)
     'güzel' (TF-IDF: 0.0239) → Benzer: bitiriyor(0.526), çerisinde(0.520), öyledir(0.517)
     'iyi' (TF-IDF: 0.0235) → Benzer: hatrı(0.475), diyebilirdim(0.467), romatik(0.466)
     📈 TF-IDF top 10'dan 4 tanesi Word2Vec'te bulundu
  🎨 İyileştirilmiş özelliklerle vektörleştirme...
  🎯 Model Performansları:
    LogisticRegression: Acc=0.5887, F1=0.6058 ❌
    RandomForest: Acc=0.6497, F1=0.5746 ❌
    SVM: Acc=0.5813, F1=0.5998 ❌
  💾 Model kaydedildi: sentiment_w2v_200d_enhanced.joblib
    Hatalı tahmin: 1051/3000

============================================================
📋 SONUÇLAR ÖZETİ
============================================================

Vektör boyutu 100:
  📊 Eğitim token: 397464, W2V kelime: 25822
  LogisticRegression: Acc=0.5873, F1=0.6048 ❌
  RandomForest: Acc=0.6517, F1=0.5819 ❌
  SVM: Acc=0.5783, F1=0.5975 ❌

Vektör boyutu 150:
  📊 Eğitim token: 397464, W2V kelime: 25822
  LogisticRegression: Acc=0.5863, F1=0.6044 ❌
  RandomForest: Acc=0.6533, F1=0.5828 ❌
  SVM: Acc=0.5780, F1=0.5978 ❌

Vektör boyutu 200:
  📊 Eğitim token: 397464, W2V kelime: 25822
  LogisticRegression: Acc=0.5887, F1=0.6058 ❌
  RandomForest: Acc=0.6497, F1=0.5746 ❌
  SVM: Acc=0.5813, F1=0.5998 ❌

🏆 EN İYİ SONUÇ:
  Accuracy: 0.6533
  Vektör boyutu: 150
  Sınıflandırıcı: RandomForest

📈 TF-IDF ÖZETİ:
  En yüksek 5 TF-IDF skoru:
    bir: 0.0652
    film: 0.0458
    ve: 0.0401
    çok: 0.0384
    bu: 0.0355

🎉 İyileştirmeler:
  ✓ Gelişmiş metin temizleme
  ✓ Sentiment-aware tokenizasyon
  ✓ Ağırlıklı Word2Vec vektörleri
  ✓ Ek metin özellikleri
  ✓ Dengelenmiş sınıflandırıcılar

```
### Güçlü Yönler
- Türkçe dil desteği ve özelleştirilmiş preprocessing
- Çoklu algoritma karşılaştırması
- Word2Vec-TF-IDF entegrasyonu
- Kapsamlı performans analizi

### Geliştirme Alanları
- Sınıf dengesizliği problemi
- OOV (Out-of-Vocabulary) kelime yönetimi
- Sentiment kelime kayıpları
- Özellik mühendisliği eksiklikleri

### Öneriler
1. **Veri Dengeleme**: SMOTE, undersampling teknikleri
2. **Model Geliştirme**: BERT, transformer modelleri
3. **Özellik Artırma**: N-gram, POS tagging
4. **Ensemble Yöntemler**: Model kombinasyonları

