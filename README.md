# Klasifikasi Penyakit Kulit MonkeyPox Menggunakan Perbandingan 3 Algoritma Deep Learning

Repositori ini berisi *source code*, hasil visualisasi, dan laporan perbandingan pengembangan 3 algoritma *Deep Learning* untuk mengklasifikasikan citra lesi kulit **Monkeypox** beserta 5 kondisi kulit lainnya menggunakan **Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0)**.

---

## 1. Latar Belakang & Tujuan
Penularan kasus Monkeypox mengalami peningkatan signifikan di berbagai negara. Untuk membantu proses identifikasi dini (*early warning*) terhadap *carrier* secara efisien tanpa bergantung penuh pada pengamatan manual ahli secara terus-menerus, dikembangkan sistem klasifikasi citra kulit berbasis *Convolutional Neural Network (CNN)* dan *Transfer Learning* yang dievaluasi menggunakan skema **5-Fold Cross Validation**.

---

## 2. Deskripsi Dataset
Dataset yang digunakan adalah **[Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0)](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20/data)** yang terbagi menjadi **6 kelas**:
1. `Chickenpox` (Cacar Air)
2. `Cowpox` (Cacar Sapi)
3. `Healthy` (Kulit Sehat)
4. `HFMD` (*Hand, Foot, and Mouth Disease*)
5. `Measles` (Campak)
6. `Monkeypox` (Cacar Monyet)

### Pembagian Data (5-Fold Cross Validation)
Sesuai ketentuan tugas:
- **Data Training**: Menggabungkan **Original Images (`Train`)** + **Augmented Images (`Train`)** pada setiap fold.
- **Data Validation**: Menggunakan **Original Images (`Valid`)** pada setiap fold.
- **Data Testing**: Menggunakan **Original Images (`Test`)** pada setiap fold (tidak pernah digunakan pada proses training).

| Fold | Data Train (Orig + Aug) | Data Valid (Orig) | Data Test (Orig) |
|:----:|:-----------------------:|:-----------------:|:----------------:|
| **Fold 1** | 8.055 citra | 144 citra | 74 citra |
| **Fold 2** | 7.635 citra | 164 citra | 82 citra |
| **Fold 3** | 8.070 citra | 153 citra | 64 citra |
| **Fold 4** | 7.860 citra | 150 citra | 81 citra |
| **Fold 5** | 8.070 citra | 154 citra | 63 citra |

---

## 3. Metode Pre-processing & 3 Algoritma yang Dibandingkan

### A. Pre-processing Citra (`Utils/getData.py`)
Seluruh citra melalui tahapan pre-processing berikut sebelum masuk ke model:
1. **Konversi Ruang Warna RGB**: Memastikan seluruh citra memiliki 3 kanal warna (`R, G, B`).
2. **Peningkatan Kontras Otomatis (`ImageOps.autocontrast`)**: Menajamkan perbedaan warna antara lesi kulit dengan area kulit sehat di sekitarnya melalui *histogram stretching*.
3. **Resizing**: Menyeragamkan ukuran citra masukan (`64 × 64` piksel untuk `SimpleCNN` dan `ModifiedCNN`, serta `128 × 128` piksel dengan normalisasi ImageNet untuk `MobileNetV3`).
4. **Normalisasi Piksel (`[0, 1]`)**: Menskalakan intensitas piksel dengan membagi `255.0` dan mengubah dimensi tensor menjadi `(C, H, W)`.
5. **One-Hot Encoding Label**: Mengubah label 6 kelas menjadi vektor biner berukuran 6.

### B. Tiga Algoritma Deep Learning (`Models/CNN.py`)
1. **`SimpleCNN` (Arsitektur 2× ConvBlock)**:
   - 2 blok konvolusi (`Conv2d(3→20, k=5)` + `ReLU` + `MaxPool2d(3)` → `Conv2d(20→50, k=5)` + `ReLU` + `MaxPool2d(3)`) + `Linear(1250→6)`.
   - Loss: `CrossEntropyLoss()` | Optimizer: `Adam(lr=0.001)` | 40 Epoch.
2. **`ModifiedCNN` (Modifikasi SimpleCNN + Antisipasi *Imbalanced Data*)**:
   - Dikembangkan menjadi 3 blok konvolusi (`32 → 64 → 128`, kernel `3×3`, `padding=1`) yang masing-masing dilengkapi `BatchNorm2d`, `ReLU`, `Dropout2d` (`0.2 / 0.3 / 0.4`), `MaxPool2d` / `AdaptiveAvgPool2d((2, 2))`, serta *classifier* `Linear(512→128)` + `BatchNorm1d` + `Dropout(0.5)` + `Linear(128→6)`.
   - **Antisipasi Data Tidak Seimbang (*Class Weights*)**: Menghitung bobot kelas berbanding terbalik dengan frekuensi sampel di tiap kelas dan menerapkannya pada `CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)` agar kelas minoritas (seperti `Measles` dan `Cowpox`) tidak terabaikan oleh kelas mayoritas (`Monkeypox`).
   - Optimizer: `Adam(lr=0.00015, weight_decay=1e-2)` + *Gaussian noise* ringan (`0.03`) untuk mencegah *overfitting*.
3. **`MobileNetV3` (`ExternalDL` — Model Deep Learning dari Luar)**:
   - Menggunakan arsitektur *Pretrained* `MobileNetV3-Small` (`IMAGENET1K_V1`) dari `torchvision.models` pada resolusi `128 × 128` dengan normalisasi ImageNet dan *custom classifier head* (`Dropout(0.5) → Linear(576→256) → Hardswish → Dropout(0.5) → Linear(256→6)`).
   - Dilengkapi *Class Weights* (`CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)`), `AdamW(lr=0.0001, weight_decay=2e-2)`, `CosineAnnealingLR`, serta augmentasi acak (*Random Flip* & *90° Rotation*).

---

## 4. Hasil Eksperimen & Perbandingan Ketiga Algoritma

### A. Tabel Perbandingan Rerata 5-Fold Cross Validation (`test.py`)

| Algoritma | Mean Accuracy | Mean Precision | Mean Recall | Mean F1-Score | Mean ROC-AUC |
|:----------|:-------------:|:--------------:|:-----------:|:-------------:|:------------:|
| **`SimpleCNN`** | 0.4506 (± 0.0579) | 0.4465 (± 0.0880) | 0.4506 (± 0.0579) | 0.3859 (± 0.0630) | 0.7623 (± 0.0180) |
| **`ModifiedCNN`** | 0.4930 (± 0.0469) | 0.5149 (± 0.0418) | 0.4930 (± 0.0469) | 0.4925 (± 0.0424) | 0.7911 (± 0.0321) |
| **`MobileNetV3`** | **0.6394 (± 0.0635)** | **0.6622 (± 0.0489)** | **0.6394 (± 0.0635)** | **0.6322 (± 0.0588)** | **0.8857 (± 0.0267)** |

> **Kesimpulan Perbandingan:**
> - **Algoritma paling akurat adalah `MobileNetV3`** dengan nilai tertinggi pada seluruh metrik evaluasi (**Mean Accuracy = `0.6394`**, **Mean Precision = `0.6622`**, **Mean F1-Score = `0.6322`**, dan **Mean ROC-AUC = `0.8857`**). Pada Fold 3, `MobileNetV3` mencapai akurasi **`0.7500` (75,00%)** dan ROC-AUC **`0.9159`**, serta **`0.9190`** pada Fold 5.
> - Modifikasi pada **`ModifiedCNN`** dengan penambahan blok konvolusi, *BatchNorm*, *Dropout*, dan *Class Weights* berhasil meningkatkan **Accuracy** dari `0.4506` menjadi **`0.4930`**, meningkatkan **F1-Score** secara signifikan dari `0.3859` menjadi **`0.4925` (+10,66%)**, serta menaikkan **ROC-AUC** menjadi **`0.7911`** dibanding `SimpleCNN` biasa.

---

### B. Detail Hasil Evaluasi per Fold

#### 1. `SimpleCNN`
| Fold | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:----:|:--------:|:---------:|:------:|:--------:|:-------:|
| Fold 1 | 0.4730 | 0.4229 | 0.4730 | 0.3790 | 0.7659 |
| Fold 2 | 0.5244 | 0.5091 | 0.5244 | 0.4865 | 0.7618 |
| Fold 3 | 0.4375 | 0.3975 | 0.4375 | 0.3760 | 0.7289 |
| Fold 4 | 0.4691 | 0.5777 | 0.4691 | 0.3989 | 0.7733 |
| Fold 5 | 0.3492 | 0.3254 | 0.3492 | 0.2890 | 0.7814 |
| **Mean ± Std** | **0.4506 ± 0.0579** | **0.4465 ± 0.0880** | **0.4506 ± 0.0579** | **0.3859 ± 0.0630** | **0.7623 ± 0.0180** |

#### 2. `ModifiedCNN`
| Fold | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:----:|:--------:|:---------:|:------:|:--------:|:-------:|
| Fold 1 | 0.4730 | 0.4737 | 0.4730 | 0.4632 | 0.7789 |
| Fold 2 | 0.4268 | 0.5044 | 0.4268 | 0.4485 | 0.7495 |
| Fold 3 | 0.5000 | 0.5263 | 0.5000 | 0.5066 | 0.7935 |
| Fold 4 | 0.4938 | 0.4804 | 0.4938 | 0.4763 | 0.7854 |
| Fold 5 | 0.5714 | 0.5897 | 0.5714 | 0.5681 | 0.8480 |
| **Mean ± Std** | **0.4930 ± 0.0469** | **0.5149 ± 0.0418** | **0.4930 ± 0.0469** | **0.4925 ± 0.0424** | **0.7911 ± 0.0321** |

#### 3. `MobileNetV3`
| Fold | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:----:|:--------:|:---------:|:------:|:--------:|:-------:|
| Fold 1 | 0.5811 | 0.6362 | 0.5811 | 0.5863 | 0.8544 |
| Fold 2 | 0.5732 | 0.5962 | 0.5732 | 0.5624 | 0.8651 |
| Fold 3 | 0.7500 | 0.7451 | 0.7500 | 0.7330 | 0.9159 |
| Fold 4 | 0.6420 | 0.6699 | 0.6420 | 0.6447 | 0.8743 |
| Fold 5 | 0.6508 | 0.6637 | 0.6508 | 0.6345 | 0.9190 |
| **Mean ± Std** | **0.6394 ± 0.0635** | **0.6622 ± 0.0489** | **0.6394 ± 0.0635** | **0.6322 ± 0.0588** | **0.8857 ± 0.0267** |

---

### C. Averaged Confusion Matrix (5-Fold Test)

| `SimpleCNN` | `ModifiedCNN` | `MobileNetV3` |
|:---:|:---:|:---:|
| ![CM SimpleCNN](results/confusion_matrix_SimpleCNN.png) | ![CM ModifiedCNN](results/confusion_matrix_ModifiedCNN.png) | ![CM MobileNetV3](results/confusion_matrix_MobileNetV3.png) |

---

### D. Grafik Training vs Validation Loss (Fold 1 - Fold 5)

#### `SimpleCNN`
| Fold 1 | Fold 2 | Fold 3 |
|:---:|:---:|:---:|
| ![SimpleCNN F1](results/training_SimpleCNN_fold1.png) | ![SimpleCNN F2](results/training_SimpleCNN_fold2.png) | ![SimpleCNN F3](results/training_SimpleCNN_fold3.png) |

| Fold 4 | Fold 5 |
|:---:|:---:|
| ![SimpleCNN F4](results/training_SimpleCNN_fold4.png) | ![SimpleCNN F5](results/training_SimpleCNN_fold5.png) |

#### `ModifiedCNN`
| Fold 1 | Fold 2 | Fold 3 |
|:---:|:---:|:---:|
| ![ModifiedCNN F1](results/training_ModifiedCNN_fold1.png) | ![ModifiedCNN F2](results/training_ModifiedCNN_fold2.png) | ![ModifiedCNN F3](results/training_ModifiedCNN_fold3.png) |

| Fold 4 | Fold 5 |
|:---:|:---:|
| ![ModifiedCNN F4](results/training_ModifiedCNN_fold4.png) | ![ModifiedCNN F5](results/training_ModifiedCNN_fold5.png) |

#### `MobileNetV3`
| Fold 1 | Fold 2 | Fold 3 |
|:---:|:---:|:---:|
| ![MobileNetV3 F1](results/training_MobileNetV3_fold1.png) | ![MobileNetV3 F2](results/training_MobileNetV3_fold2.png) | ![MobileNetV3 F3](results/training_MobileNetV3_fold3.png) |

| Fold 4 | Fold 5 |
|:---:|:---:|
| ![MobileNetV3 F4](results/training_MobileNetV3_fold4.png) | ![MobileNetV3 F5](results/training_MobileNetV3_fold5.png) |

---

### E. Kurva ROC-AUC (Fold 1 - Fold 5)

#### `SimpleCNN`
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|:---:|:---:|:---:|:---:|:---:|
| ![ROC SimpleCNN F1](results/roc_SimpleCNN_fold1.png) | ![ROC SimpleCNN F2](results/roc_SimpleCNN_fold2.png) | ![ROC SimpleCNN F3](results/roc_SimpleCNN_fold3.png) | ![ROC SimpleCNN F4](results/roc_SimpleCNN_fold4.png) | ![ROC SimpleCNN F5](results/roc_SimpleCNN_fold5.png) |

#### `ModifiedCNN`
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|:---:|:---:|:---:|:---:|:---:|
| ![ROC ModifiedCNN F1](results/roc_ModifiedCNN_fold1.png) | ![ROC ModifiedCNN F2](results/roc_ModifiedCNN_fold2.png) | ![ROC ModifiedCNN F3](results/roc_ModifiedCNN_fold3.png) | ![ROC ModifiedCNN F4](results/roc_ModifiedCNN_fold4.png) | ![ROC ModifiedCNN F5](results/roc_ModifiedCNN_fold5.png) |

#### `MobileNetV3`
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|:---:|:---:|:---:|:---:|:---:|
| ![ROC MobileNetV3 F1](results/roc_MobileNetV3_fold1.png) | ![ROC MobileNetV3 F2](results/roc_MobileNetV3_fold2.png) | ![ROC MobileNetV3 F3](results/roc_MobileNetV3_fold3.png) | ![ROC MobileNetV3 F4](results/roc_MobileNetV3_fold4.png) | ![ROC MobileNetV3 F5](results/roc_MobileNetV3_fold5.png) |

---

## 5. Struktur Repositori
```text
├── Models/
│   └── CNN.py          # 3 Arsitektur model (SimpleCNN, ModifiedCNN, ExternalDL/MobileNetV3)
├── Utils/
│   └── getData.py      # Dataset loader & pre-processing citra (RGB, Autocontrast, Resize, Normalisasi)
├── results/            # Grafik Loss, Averaged Confusion Matrix, dan ROC Curve (Fold 1 - 5)
├── train.py            # Script pelatihan 5-Fold Cross Validation untuk ketiga algoritma
├── test.py             # Script pengujian, evaluasi metrik 5-Fold, dan tabel perbandingan
└── README.md           # Laporan lengkap hasil pengembangan model
```

## 6. Cara Menjalankan Kode
1. Ekstrak dataset MSLD v2.0 ke dalam folder `Mpox_Dataset/`.
2. Install *dependencies* yang diperlukan:
   ```bash
   pip install torch torchvision scikit-learn matplotlib seaborn pandas pillow
   ```
3. Jalankan proses pelatihan 5-Fold untuk ketiga algoritma:
   ```bash
   python train.py
   ```
4. Jalankan proses evaluasi & perbandingan pada data *Test*:
   ```bash
   python test.py
   ```
