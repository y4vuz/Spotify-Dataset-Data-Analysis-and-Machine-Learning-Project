# Spotify-Dataset-Data-Analysis-and-Machine-Learning-Project
# 🎵 End-to-End Spotify Popularity Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Machine_Learning-Scikit--Learn-orange)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Data_Analysis-Pandas-150458)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Visualization-Seaborn-4ebd9e)](https://seaborn.pydata.org/)

Bu proje, Spotify müzik veri setini kullanarak bir şarkının popülerliğini tahmin eden uçtan uca (end-to-end) bir makine öğrenmesi pipeline'ıdır. Çalışma; veri ön işleme, detaylı keşifsel veri analizi (EDA), özellik mühendisliği (feature engineering) ve iki farklı yaklaşımla (Regresyon ve Sınıflandırma) modelleme aşamalarını içermektedir.

## 📌 Proje Özeti
* **Amaç:** Şarkıların teknik ses özelliklerini (dans edilebilirlik, enerji, tempo vb.) ve sanatçı geçmişini kullanarak popülerlik skorunu (0-100) ve popüler olup/olmama durumunu tahmin etmek.
* **Veri Seti:**  [Veri Seti](https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv)
* **Algoritmalar:** Random Forest Regressor & Random Forest Classifier

## 🛠️ Veri Ön İşleme ve Özellik Mühendisliği (Feature Engineering)
Modelin performansını maksimize etmek ve veri bütünlüğünü korumak için kritik adımlar atılmıştır:

1. **Tip Senkronizasyonu (Leading Zeros Fix):** `track_id`, `track_album_id` ve `playlist_id` gibi başında sıfır bulunan veriler `string` olarak parse edilerek veri kaybı (sekizlik/octal okuma hatası) engellenmiştir.
2. **Kategorik Veri İmputasyonu:** Eksik veriler frekansı en yüksek olan (Mode) değerlerle doldurulmuştur.
3. **Yeni Öznitelik Üretimi (`artist_avg_popularity`):** Şarkı popülerliğindeki en büyük etkenin sanatçının genel popülaritesi olduğu hipotezinden yola çıkılarak, sanatçıların ortalama popülerlik skorları hesaplanmış ve modele yeni bir öznitelik olarak beslenmiştir. Bu adım modelin açıklanabilirliğini (R²) kritik ölçüde artırmıştır.
4. **Encoding & Scaling:** Sayısal veriler `StandardScaler` ile ölçeklendirilmiş, kategorik veriler (`playlist_genre`, vb.) `OneHotEncoder` kullanılarak Pipeline içerisine entegre edilmiştir.

## 📊 Keşifsel Veri Analizi (EDA)
Veri setindeki gizli örüntüleri ortaya çıkarmak için Seaborn pastel renk paletiyle çeşitli görselleştirmeler yapılmıştır:
* **Müzik Türü Dağılımı:** Veri setindeki şarkıların türlere göre yüzdesel dağılımını gösteren pasta grafiği (`07_genre_distribution_pie_chart.png`).
* **Korelasyon Analizleri:** Sanatçı popülerliği vs. Şarkı Popülerliği ve Ses Yüksekliği (Loudness) vs. Şarkı Popülerliği dağılım grafikleri.
* **Feature Importance:** Random Forest modeline göre popülerliği etkileyen en önemli 10 özelliğin (Top 10) görselleştirilmesi.

## ⚙️ Modelleme ve Metrikler

Projede problemi iki farklı boyutta çözmek için iki ayrı model eğitilmiştir:

### 1. Regresyon Modeli (Kesin Skor Tahmini)
* **Model:** Random Forest Regressor (`n_estimators=300`, `max_depth=20`)
* **Hedef:** 0 ile 100 arasında net bir popülerlik skoru tahmini.
* **Metrikler:** R-Squared (R²), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) hesaplanarak modelin sapma payı ölçülmüştür.

### 2. Sınıflandırma Modeli (Popülerlik Potansiyeli)
* **Model:** Random Forest Classifier (`n_estimators=150`)
* **Hedef:** Popülerlik skoru 50'den büyük olanları "Popüler (1)", küçük olanları "Popüler Değil (0)" olarak ayırmak.
* **Metrikler:** Accuracy, F1-Score, Precision değerlendirilmiş ve sonuçlar Confusion Matrix (Karmaşıklık Matrisi) ile görselleştirilmiştir.



