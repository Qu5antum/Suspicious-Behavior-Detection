# Güvenlik Kameralarında Şüpheli Davranış Tespiti

## Gerçek zamanlı video analizi ile şüpheli insan davranışlarını otomatik olarak tespit eden bilgisayarlı görü sistemi.

## Proje Hakkında

Bu proje, güvenlik kameralarından gelen canlı video akışlarını analiz ederek **şüpheli insan davranış kalıplarını** gerçek zamanlı olarak tespit etmek amacıyla geliştirilmiştir. Sistem, güvenlik personelinin manuel izleme yükünü azaltmak ve dikkat gerektiren olayları otomatik olarak işaretlemek için tasarlanmıştır.

### Tespit Edilen Şüpheli Davranışlar

Davranış

 - **Loitering**  Kişinin aynı alanda uzun süre hareketsiz kalması 
 - **Repeated Path**  Aynı güzergahı tekrar tekrar izleme 
 - **Looking Around**  Sık sık baş çevirme, etrafı gözetleme 
 - **Abandoned Object**  Çanta, valiz gibi eşyaların sahipsiz bırakılması 


## Özellikler

- Gerçek zamanlı çoklu kişi takibi ve ID ataması
- Her kişi için bağımsız davranış analizi ve risk puanlaması
- Renkli sınırlayıcı kutularla görsel uyarı sistemi (yeşil → turuncu → kırmızı)
- Sahipsiz eşya tespitinde üç seviyeli uyarı: `NORMAL` / `WARNING` / `ALERT`
- Kafa pozisyonu tahmini ile gözetleme davranışı tespiti
- Yörünge takibi ve tekrarlayan hareket analizi
- Windows MSMF hatalarına karşı otomatik kamera yeniden bağlantısı
- HUD (Heads-Up Display) ile anlık istatistikler


## Sistem Mimarisi

```
Video Akışı (Kamera / Dosya)
         │
         ▼
┌─────────────────────┐
│   YOLOv8 Deteksiyon │  ← Kişi + Eşya tespiti
└─────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│ByteTrack│ │IoU Tracker│  ← Kişi ve eşya takibi
└────────┘ └──────────┘
    │              │
    └──────┬───────┘
           ▼
┌──────────────────────┐
│  OwnershipAnalyzer   │  ← Eşya-sahip ilişkisi
└──────────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐   ┌─────────┐
│Behavior│   │Abandoned│  ← Davranış analizi
│Analyzer│   │Analyzer │
└────────┘   └─────────┘
           │
           ▼
     Görsel Çıktı
```

## Kullanılan Teknolojiler

### Nesne Tespiti
- **YOLOv8n** — Ultralytics tarafından geliştirilen hafif ve hızlı nesne tespit modeli. Hem kişi hem de eşya tespiti için kullanılır. Gerçek zamanlı performans için `yolov8n.pt`, daha yüksek doğruluk için `yolov8s.pt` veya `yolov8m.pt` modelleri tercih edilebilir.

### Kişi Takibi
- **ByteTrack** — Ultralytics'e entegre edilmiş gelişmiş çok nesne takip algoritması. Kısa süreli kaybolmalarda bile kişi ID'sini korur; bu özelliği, kişinin sahipsiz bıraktığı eşyayla ilişkilendirilmesi için kritik öneme sahiptir.

### Eşya Takibi
- **IoU Tabanlı Tracker** — Kesişim/Birleşim oranına dayalı özel eşya takipçisi. Geçici engellemeler (okluzyonlar) sırasında merkez koordinat yöntemine kıyasla daha kararlı sonuçlar üretir.

### Sahip Analizi
- **IoU Genişletilmiş BBox Eşleştirme** — Eşyanın sınırlayıcı kutusu genişletilir ve her kişinin kutusuyla IoU değeri hesaplanır. Kamera açısı ve ölçeğinden bağımsız çalışır.

### Görüntü İşleme
- **OpenCV** — Video yakalama, kare işleme ve görsel çıktı için kullanılır. DirectShow (Windows) ve V4L2 (Linux) backend desteği mevcuttur.

### Kafa Pozu Tahmini
- **OpenCV DNN** — Caffe tabanlı SSD yüz dedektörü (`res10_300x300_ssd_iter_140000.caffemodel`). Kişi bounding box'ının üst %40'lık kısmında yüz arar ve yaw açısını hesaplar.


## Davranış Analizi Detayları

### 1. Loitering (Aylak Gezme) Tespiti

Bir kişinin belirli bir bölgede uzun süre az hareketle bulunması durumunu tespit eder.

**Yöntem:**
- Her kare için kişinin merkez koordinatı hesaplanır.
- Son `N` karedeki toplam yer değiştirme mesafesi ölçülür.
- Mesafe eşik değerinin altında kalırsa `loitering` işaretlenir.

**Parametreler:**

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `loiter_threshold_px` | 30 | Hareket sayılmayan piksel değişimi |
| `loiter_frames` | 90 | Bu kadar kare hareketsiz kalırsa tespit |

---

### 2. Repeated Path (Tekrarlayan Güzergah) Tespiti

Kişinin aynı rotayı birden fazla kez izlemesi durumunu tespit eder.

**Yöntem:**
- Kişinin yörüngesi segment listesi olarak saklanır.
- Yeni segmentler geçmiş segmentlerle karşılaştırılır.
- Cosinüs benzerliği belirli eşiğin üzerindeyse `repeated_path` işaretlenir.

**Kullanım Senaryosu:** Bir güzergahı defalarca dolaşan gözetleme davranışı.

---

### 3. Looking Around (Etrafı Gözetleme) Tespiti

Kişinin kafasını sık sık sağa-sola çevirmesi davranışını tespit eder.

**Yöntem:**
1. Kişi bounding box'ının üst %40'ı yüz bölgesi olarak belirlenir.
2. OpenCV DNN ile yüz tespit edilir.
3. Yüz merkezinin baş bounding box içindeki yatay konumundan **yaw açısı** hesaplanır.
4. EMA (Üstel Hareketli Ortalama) ile gürültü filtrelenir.
5. Histerezisli otomat ile `SOL` / `MERKEZ` / `SAĞ` durumları takip edilir.
6. Yeterli amplitüdde yeterli sayıda geçiş tespit edilirse `looking_around` işaretlenir.

**Parametreler:**

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `ema_alpha` | 0.25 | EMA yumuşatma katsayısı |
| `enter_threshold` | 0.25 | Sola/sağa geçiş eşiği |
| `exit_threshold` | 0.12 | Merkeze dönüş eşiği (histerezis) |
| `amplitude_threshold` | 0.30 | Geçerli sayılan minimum dönüş genişliği |
| `min_switches` | 2 | Tespit için gereken minimum geçiş sayısı |

---

### 4. Abandoned Object (Sahipsiz Eşya) Tespiti

Bir kişinin yanında taşıdığı çanta, sırt çantası veya valizi bırakıp uzaklaşmasını tespit eder.

**Tespit Mantığı:**

```
Kişi eşyayla birlikte görünür
        │
        ▼ Kişi uzaklaşır veya kameradan çıkar
        │
        ▼ alert_frames kadar eşya hareketsiz kalır
        │
        ▼
   ⚠ ALERT durumu
```

**Üç Durum:**

| Durum | Renk | Koşul |
|---|---|---|
| `NORMAL` | Yeşil | Sahip yakında veya eşya hareket ediyor |
| `WARNING` | Turuncu | Sahip hiç görünmedi, eşya uzun süre hareketsiz |
| `ALERT` | Kırmızı | Sahip görüldü, uzaklaştı, eşya kaldı |

**IoU Tabanlı Sahip Eşleştirme:**

Eşyanın bounding box'ı `owner_expand_px` piksel genişletilir. Genişletilmiş kutu ile her kişinin kutusu arasındaki IoU hesaplanır. Bu yöntem, kamera açısı değişimlerinde merkez mesafesine kıyasla çok daha güvenilir sonuç verir.

**Parametreler:**

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `owner_expand_px` | 80 | Sahip araması için bbox genişleme miktarı |
| `owner_iou_thresh` | 0.05 | Sahip sayılmak için minimum IoU değeri |
| `alert_frames` | 50 | Sahibi gittikten sonra ALERT için gereken kare sayısı |
| `warning_frames` | 100 | Hiç sahip görünmeden WARNING için gereken kare sayısı |
| `static_px` | 6 | Hareketsiz sayılan maksimum piksel kayması |

---

## Dosya Yapısı

```
suspicious-behavior-detection/
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py              # Ana pipeline (tüm davranışlar)
│   ├── abandoned_pipeline.py    # Sadece sahipsiz eşya pipeline'ı
│   │
│   ├── person_detector.py       # PersonDetector, PersonTracker
│   ├── object_tracking.py       # BagDetector, BagTracker, OwnershipAnalyzer
│   ├── trajectory.py            # TrajectoryManager, TrajectoryAnalyzer
│   └── behavior.py              # BehaviorAnalyzer, LookingAroundAnalyzer
│
├── deploy.prototxt.txt          # Yüz dedektörü yapılandırması
├── res10_300x300_ssd_iter_140000.caffemodel  # Yüz dedektörü ağırlıkları
├── yolov8n.pt                   # YOLOv8 model ağırlıkları (otomatik indirilir)
├── requirements.txt
└── README.md
```

---

## Risk Puanlama Sistemi

Her kişiye davranışlarına göre risk puanı atanır. Puan, sınırlayıcı kutunun rengini belirler.

| Davranış | Puan |
|---|---|
| Loitering | +2 |
| Repeated Path | +3 |
| Looking Around | +2 |
| Yakınında ALERT eşyası | +4 |
| Yakınında WARNING eşyası | +1 |

**Renk Skalası:**

| Puan | Renk | Anlam |
|---|---|---|
| 0 – 2 | Yeşil | Normal |
| 3 – 5 | Turuncu | Dikkat |
| 6+ | Kırmızı | Şüpheli |

---

## Bilinen Sınırlamalar

- Yüz tespiti, kişi kameraya sırtını döndüğünde çalışmaz; bu durumda `looking_around` tespit edilemez.
- Kalabalık sahnelerde IoU tabanlı eşleştirme yanlış sahip ataması yapabilir.
- ByteTrack, 30 FPS altı video akışlarında ID tutarlılığını yitirebilir.