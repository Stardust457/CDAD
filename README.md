# CDAD: A Multi-Scale Feature Fusion Model for Cross-Domain Anomaly Detection

> **Accepted at IJCNN 2026**

---

## 📄 Abstract

Anomaly detection aims to identify deviations by learning normal data distributions, mainly covering industrial, semantic, and medical domains. Most existing methods, however, are domain-specific and degrade when transferred. To address this limitation, we proposed CDAD, a unified Cross-Domain Anomaly Detection model. CDAD leverages a pre-trained feature extractor with a feature adaptor to align cross-domain features and tighten the decision boundary of normal samples. A hybrid loss emphasizes hard-to-classify anomalies, while a Multi-Scale Feature Fusion Discriminator integrates global and local features. Extensive experiments on five datasets across three domains demonstrate that CDAD consistently outperforms recent state-of-the-art approaches, highlighting its robustness and effectiveness in cross-domain anomaly detection.

---

## 🔧 Environment Setup

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Stardust457/CDAD.git
cd CDAD
pip install -r requirements.txt
```

---

## 📦 Datasets

This project is evaluated on five datasets. Please download each dataset and place it under the appropriate directory before running.

### 1. MPDD

- **Download:** [MPDD Dataset](https://vutbr-my.sharepoint.com/:f:/g/personal/xjezek16_vutbr_cz/EhHS_ufVigxDo3MC6Lweau0BVMuoCmhMZj6ddamiQ7-FnA?e=oHKCxI)

### 2. MVTec-AD

- **Download:** [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### 3. MVTec-LOCO

- **Download:** [MVTec LOCO Dataset](https://www.mvtec.com/company/research/datasets/mvtec-loco)

### 4. Aircraft-FGVC

- This dataset is automatically downloaded via torchvision in our code, so no manual download is required.

### 5. APTOS

- **Download:** [APTOS Dataset](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)

---

## 🚀 Training & Evaluation

Each dataset has a dedicated shell script under `runs/` that handles both training and evaluation end-to-end.

### MPDD

```bash
bash runs/run_mpdd.sh
```

### MVTec-AD

```bash
bash runs/run_mvtec_ad.sh
```

### MVTec-LOCO

```bash
bash runs/run_mvtec_loco.sh
```

### Aircraft-FGVC

```bash
bash runs/run_aircraft_fgvc.sh
```

### APTOS

```bash
bash runs/run_aptos.sh
```
