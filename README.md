# Spotify-Dataset-Data-Analysis-and-Machine-Learning-Project
Bu proje, Spotify veri setindeki şarkı özelliklerini (audio features) kullanarak bir şarkının popülerlik skorunu tahmin etmeye yönelik bir makine öğrenmesi çalışmasıdır. Proje, veri temizlemeden model değerlendirmeye kadar uçtan uca bir veri bilimi sürecini kapsar.

## 📊 Veri Seti Özeti
Çalışmada kullanılan veri seti, yaklaşık 30.000 şarkıya ait teknik özellikleri içermektedir.
* **Kaynak:** Kaggle / TidyTuesday Spotify Dataset
* **Hedef Değişken:** `track_popularity` (0-100 arası skor)
* **Özellikler:** Danceability, Energy, Key, Loudness, Acousticness, Instrumentalness, Valence, Tempo, vb.

## 🛠️ Teknik Zorluklar ve Çözümler

### Veri Tipi Senkronizasyonu (Leading Zeros Sorunu)
Veri setinin yüklenmesi aşamasında, `track_id` ve `track_album_id` gibi sütunlarda bulunan ve başında "0" (sıfır) olan uzun sayı dizilerinin Python tarafından yanlışlıkla tam sayı (integer) olarak algılanması veri bozulmasına neden olmaktaydı.
* **Çözüm:** `pandas.read_csv` fonksiyonunda `dtype={'track_id': str}` parametresi kullanılarak bu kimlik numaralarının metin (string) olarak okunması sağlandı ve veri bütünlüğü korundu.

## ⚙️ Model Mimarisi
Projede, değişkenler arasındaki doğrusal olmayan karmaşık ilişkileri modellemek amacıyla **Random Forest Regressor** algoritması tercih edilmiştir.

1. **Feature Engineering:** `playlist_genre` ve `playlist_subgenre` gibi kategorik veriler Label Encoding ile sayısal forma dönüştürüldü.
2. **Preprocessing:** Modelin doğruluğunu etkileyebilecek gereksiz sütunlar (`track_name`, `track_artist` vb.) veri setinden çıkarıldı.
3. **Train-Test Split:** Veri seti %80 eğitim ve %20 test olarak ayrıldı.
