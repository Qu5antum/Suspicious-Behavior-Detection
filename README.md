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


