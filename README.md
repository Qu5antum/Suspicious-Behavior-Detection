# CCTV'de Şüpheli Davranış Tespiti(Suspicious Behavior Detection)

## Amaç

Bu proje, güvenlik kameralarından gelen video akışlarını analiz eden ve **şüpheli insan davranış kalıplarını** otomatik olarak tespit eden bir sistem geliştirmeyi hedefler.

Örnek şüpheli davranışlar:

- Uzun süre aynı alanda bulunma (**loitering**)  
- Aynı güzergah boyunca tekrarlanan hareket (**patrol / repeated path**)  
- Sık sık baş çevirme (**looking around**)  
- Bir nesneyi bırakma (**abandoned object**)  

---

## Kullanılan Teknolojiler

- **Nesne Algılama:** YOLOv8  
- **Nesne Takibi:** DeepSORT  
- **Computer Vision:** OpenCV  
- **Deep Learning:** PyTorch  
- **Davranış Analizi:** Custom trajectory ve behavior modelleri  

---

## Davranış Analizi Detayları

### 1. Loitering Detection

Kişi aynı bölgede belirlenen süre boyunca az hareket ediyorsa tespit edilir.

- **Yöntem:**  
  - Sınır kutusunun merkezini hesapla  
  - Koordinatlardaki değişiklikleri kontrol et  

### 2. Repeated Path

- Bir kişinin hareket yörüngesini izler.

- Bir kişi aynı yolu tekrar tekrar izlerse, davranış `tekrarlanan_yol` olarak işaretlenir.

- Daha yüksek **şüpheli puan** ve turuncu/kırmızı sınırlayıcı kutularla görselleştirilir.

### 3. Looking Around

- Kişinin kafa pozisyonunu, sınırlayıcı kutunun üst kısmını veya tespit edilen yüzü kullanarak tahmin eder.

- Kişi sık sık başını sağa sola çeviriyorsa, `etrafına bakıyor` olarak işaretlenir.

- Bölgeyi gözetleyen kişileri tespit etmek için kullanışlıdır.

- Daha yüksek bir **şüpheli puanı** ile görselleştirilir ve sınırlayıcı kutunun rengi değişir.

