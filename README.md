# Metal Yüzeylerde Kusur Tespiti

Klasik görüntü işleme yöntemleriyle metal somun görüntülerinde kusurlu bölgelerin bulunması,
kusur alanının hesaplanması ve **Geçti / Hurda** kalite kararı verilmesi.

BYM445 Görüntü İşleme dersi dönem projesi (Konu 9: Metal Yüzeylerde Çizik ve Hata Tespiti).

## Yaklaşım

Proje iki klasik işlem hattını aynı test görüntüleri üzerinde karşılaştırır:

| | İşlem hattı |
|---|---|
| **Yöntem A** | Ön işleme → Canny kenar tespiti → morfolojik işlemler |
| **Yöntem B** | Ön işleme → Gabor / black-hat filtreleme → eşikleme → morfolojik işlemler |

Bulunan kusur maskeleri veri setindeki referans maskelerle karşılaştırılır:
piksel düzeyinde IoU ve Dice, görüntü düzeyinde Precision, Recall, F1 ve karmaşıklık matrisi.

## Veri seti

[MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) veri setinin `metal_nut` kategorisi kullanılır
(Bergmann vd., 2019; lisans: CC BY-NC-SA 4.0). Veri seti boyutu nedeniyle (~5 GB) depoya eklenmemiştir.

1. Veri setini MVTec'in resmî sayfasından ya da aynı arşivi barındıran
   [Hugging Face kopyasından](https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection) indirin
   (`mvtec_anomaly_detection.tar.xz`).
2. Arşivi `data/mvtec_ad/` klasörüne açın. Beklenen yapı:

```
data/mvtec_ad/metal_nut/
├── train/good/            220 sağlam görüntü
├── test/{good,scratch,bent,color,flip}/
└── ground_truth/{scratch,bent,color,flip}/   referans maskeler
```

## Kurulum

Python 3.12 ile:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
jupyter notebook
```

## Klasör yapısı

```
├── src/          yeniden kullanılan kod (veri okuma, yöntemler, ölçütler)
├── notebooks/    adım adım deneyler
├── results/      üretilen tablolar ve görseller
└── docs/         proje raporları
```

## Defterler

| Defter | İçerik |
|---|---|
| [01_veri_inceleme](notebooks/01_veri_inceleme.ipynb) | Görüntü sayıları, kusur türleri, maske alanları, somunların konum ve parlaklık özellikleri |

## Kaynaklar

- Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). MVTec AD — A comprehensive real-world dataset for unsupervised anomaly detection. *CVPR*.
- Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., & Steger, C. (2021). The MVTec anomaly detection dataset. *International Journal of Computer Vision*, 129, 1038–1059.
