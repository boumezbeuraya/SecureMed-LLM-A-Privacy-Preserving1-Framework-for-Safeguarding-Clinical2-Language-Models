<div align="center">

<img src="Graphical-Abstract.PNG" alt="SecureMed-LLM Architecture" width="800"/>

# SecureMed-LLM

### A Privacy-Preserving Framework for Safeguarding Clinical Language Models

[![Paper](https://img.shields.io/badge/Paper-PeerJ%20JCS-blue?style=flat-square&logo=readthedocs)](https://peerj.com/computer-science/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Opacus](https://img.shields.io/badge/Differential%20Privacy-Opacus-purple?style=flat-square)](https://opacus.ai/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow?style=flat-square)](https://huggingface.co/)
[![Dataset](https://img.shields.io/badge/Dataset-Open--I%20Chest%20X--ray-orange?style=flat-square&logo=kaggle)](https://www.kaggle.com/datasets/financekim/curated-cxr-report-generation-dataset/data)

> **Official implementation** of *"SecureMed-LLM: A Privacy-Preserving Framework for Safeguarding Clinical Language Models"*
> — Aya Boumezbeur, Fouzi Harrag, Mohamed Deriche, Muhammad Khan
> — Submitted to *PeerJ Computer Science*, November 2025

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Motivation](#-motivation)
- [Key Contributions](#-key-contributions)
- [Framework Architecture](#-framework-architecture)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Usage](#-usage)
  - [Data Preprocessing & Anonymization](#1-data-preprocessing--phi-anonymization)
  - [Privacy-Preserving Fine-Tuning](#2-privacy-preserving-fine-tuning)
  - [Adversarial Training](#3-adversarial-training)
  - [IDS-LLM Validation](#4-ids-llm-validation)
  - [Encrypted Inference](#5-encrypted-inference)
  - [Full Pipeline](#6-running-the-full-pipeline)
- [Experimental Results](#-experimental-results)
- [Configuration](#-configuration)
- [Citation](#-citation)
- [License](#-license)
- [Contact](#-contact)

---

## 🧠 Overview

**SecureMed-LLM** is a modular, multi-layered safeguarding framework for privacy-preserving clinical report generation. It addresses three critical threat classes simultaneously:

| Threat Class | Risk | Defense |
|---|---|---|
| **Privacy Leakage** | Membership inference, PHI exposure | DP-SGD + Med-Guard anonymization |
| **Adversarial Manipulation** | Image/text perturbations, prompt injection | Adversarial fine-tuning (FGSM, PGD, DeepFool) |
| **Unsafe Outputs** | Hallucinated findings, logical inconsistencies | IDS-LLM rule-based + anomaly detection |
| **Data Interception** | Report eavesdropping, unauthorized access | ECIES/Curve25519 encrypted delivery |

Unlike existing approaches that treat these threats in isolation, SecureMed-LLM integrates all defenses into a **coordinated end-to-end pipeline** spanning training, inference, and deployment.

---

## 💡 Motivation

Large Language Models are rapidly entering clinical workflows — radiology reporting, triage support, patient advisory — yet expose serious attack surfaces:

- A subtly perturbed chest X-ray can cause an LLM to generate a report *inverting* a pneumonia finding
- Prompt injection attacks can downgrade the urgency of critical triage cases
- Membership inference attacks can recover sensitive patient data from model confidence scores
- Unencrypted report transmission enables interception by hospital network adversaries

These risks are especially severe in **resource-constrained and developing-world healthcare settings**, where compromised AI systems can directly exacerbate care inequities.

---

## 🏆 Key Contributions

1. **Comprehensive Vulnerability Analysis** — Systematic categorization of LLM security risks in high-stakes clinical settings with clinically relevant failure modes.

2. **Structured Defense Survey** — Critical review of hybrid neural-symbolic, reasoning-augmented, privacy-preserving, threat-specific, and domain-specific safeguards.

3. **SecureMed-LLM Framework** — A multi-layered defense architecture combining:
   - Local multimodal PHI anonymization (Med-Guard)
   - DP-SGD training with adversarial augmentation
   - IDS-LLM post-generation validation
   - ECIES encrypted report delivery

4. **Empirical Evaluation** — Evaluation on Open-I chest X-ray dataset across adversarial attacks (FGSM, PGD, DeepFool), membership inference attacks, and PHI leakage, with ablation studies for each component.

5. **Deployment Insights** — Practical analysis of inference latency, clinical feasibility, and resource-constrained deployment.

---

## 🏗 Framework Architecture

SecureMed-LLM operates through **six sequential security levels**:

```
Clinician Device                          Secure Hospital Server
─────────────────────────────────         ─────────────────────────────────────────
                                          
 Level 1: Raw Input                       Level 4: Privacy-Preserving Generation
 ┌──────────────────────┐                 ┌────────────────────────────────────────┐
 │  Chest X-ray Image   │                 │  BioMedCLIP Encoder                    │
 │  + Clinical Notes    │                 │  → Linear Projection                   │
 └──────────┬───────────┘                 │  → T5 Decoder (DP-SGD trained, ε=3.0)  │
            │                             │  + Adversarial augmentation (5% ratio) │
 Level 2: Med-Guard Anonymization         └──────────────┬─────────────────────────┘
 ┌──────────▼───────────┐                                │
 │  Text: Presidio NER  │                 Level 5: IDS-LLM Validation
 │  (PHI → [MASK])      │                 ┌──────────────▼─────────────────────────┐
 │  Image: Gaussian     │                 │  Rule-Based Filtering (47 rules)       │
 │  Noise σ=15          │                 │  Clinical Entity Verification          │
 └──────────┬───────────┘                 │  Semantic Anomaly Detection            │
            │                             │  (Sentence-BERT + Isolation Forest)    │
 Level 3: Secure Transmission            └──────────────┬─────────────────────────┘
 ┌──────────▼───────────┐                                │
 │  TLS 1.3 / HTTPS     │                 Level 6: Encrypted Delivery
 └──────────────────────┘                 ┌──────────────▼─────────────────────────┐
                                          │  ECIES / Curve25519 Encryption         │
                                          │  Physician Public Key                  │
                                          └────────────────────────────────────────┘
```

### Core Models

| Component | Model | Role |
|---|---|---|
| Vision Encoder | [BioMedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | Radiological image feature extraction |
| Text Decoder | [T5-base](https://huggingface.co/t5-base) | Structured clinical report generation |
| Adversarial Generator | [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) | BioMedAttack-LLM for adversarial sample generation |
| Anomaly Detector | Sentence-BERT + Isolation Forest | Semantic deviation detection in generated reports |

---

## 📂 Repository Structure

```
SecureMed-LLM/
│
├── README.md                        # This file
├── LICENSE                          # MIT License
├── requirements.txt                 # Python dependencies (pinned)
├── environment.yml                  # Conda environment specification
├── Graphical-Abstract.PNG           # Architecture overview figure
│
├── configs/
│   └── config.yaml                  # All hyperparameters and paths
│
├── src/
│   ├── anonymization/
│   │   ├── medguard_text.py         # Presidio-based PHI text de-identification
│   │   └── medguard_image.py        # Gaussian noise image anonymization
│   │
│   ├── models/
│   │   ├── biomed_clip_encoder.py   # BioMedCLIP vision encoder wrapper
│   │   ├── t5_decoder.py            # T5 text decoder with projection layer
│   │   └── pipeline.py             # Full BioMedCLIP + T5 pipeline
│   │
│   ├── privacy/
│   │   └── dp_training.py           # DP-SGD training with Opacus
│   │
│   ├── adversarial/
│   │   ├── attack_generator.py      # BioMedAttack-LLM adversarial sample generation
│   │   └── adv_training.py          # Adversarial fine-tuning (FGSM, PGD, DeepFool)
│   │
│   ├── validation/
│   │   └── ids_llm.py               # IDS-LLM: rule-based + clinical + anomaly detection
│   │
│   ├── encryption/
│   │   ├── encrypt_report.py        # ECIES/Curve25519 report encryption
│   │   └── decrypt_report.py        # Local decryption by authorized physician
│   │
│   └── utils/
│       ├── metrics.py               # BLEU, SSIM, PSNR, MIA accuracy computation
│       └── data_loader.py           # Open-I dataset loading and preprocessing
│
├── scripts/
│   ├── train.py                     # Full training pipeline entry point
│   ├── inference.py                 # Inference / report generation entry point
│   └── evaluate.py                  # Evaluation and metrics reporting
│
└── docs/
    └── ARCHITECTURE.md              # Detailed architecture documentation
```

---

## ⚙️ Installation

### Option A — pip (recommended for quick setup)

```bash
# Clone the repository
git clone https://github.com/ayamouna/SecureMed-LLM-A-Privacy-Preserving1-Framework-for-Safeguarding-Clinical2-Language-Models.git
cd SecureMed-LLM-A-Privacy-Preserving1-Framework-for-Safeguarding-Clinical2-Language-Models

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option B — Conda

```bash
conda env create -f environment.yml
conda activate securemed-llm
```

### Requirements

- Python ≥ 3.8
- CUDA-compatible GPU recommended for training (NVIDIA T4/P100 or equivalent)
- CPU-only mode supported for inference (~6.5 s/report on Intel Xeon 2.0 GHz)

---

## 📊 Dataset

This project uses the **Open-I Chest X-ray Dataset** (curated NIH Open-I corpus, Kaggle version).

| Split | Pairs | Description |
|---|---|---|
| Training | 93,347 | Original image–report pairs (unmodified) |
| Validation | 1,885 | Original pairs, used for noise level and threshold tuning |
| Test | 1,541 | Original pairs, used for all reported metrics |

- Each entry: a chest X-ray image + paired radiology report (Findings + Impression sections)
- Average report length: ~56 words (range: 30–70 words)
- Source institution: Indiana University (single-site; see [Limitations](#limitations))

**Download:**
```bash
# Kaggle CLI
kaggle datasets download -d financekim/curated-cxr-report-generation-dataset
unzip curated-cxr-report-generation-dataset.zip -d data/open-i/
```
Or download manually from: https://www.kaggle.com/datasets/financekim/curated-cxr-report-generation-dataset/data

---

## 🚀 Usage

All scripts accept a `--config` argument pointing to `configs/config.yaml`. Override any parameter with `--key value` flags.

### 1. Data Preprocessing & PHI Anonymization

```bash
# Text de-identification (Presidio NER)
python src/anonymization/medguard_text.py \
    --input_dir data/open-i/reports \
    --output_dir data/anonymized/reports

# Image anonymization (Gaussian noise, sigma=15)
python src/anonymization/medguard_image.py \
    --input_dir data/open-i/images \
    --output_dir data/anonymized/images \
    --sigma 15
```

### 2. Privacy-Preserving Fine-Tuning

```bash
python scripts/train.py \
    --mode dp_finetune \
    --data_dir data/anonymized \
    --output_dir checkpoints/dp_model \
    --epsilon 3.0 \
    --delta 1e-5 \
    --noise_multiplier 1.1 \
    --clip_norm 1.0 \
    --epochs 5 \
    --batch_size 16 \
    --lr 2e-5
```

### 3. Adversarial Training

```bash
# Generate adversarial samples with BioMedAttack-LLM
python src/adversarial/attack_generator.py \
    --data_dir data/anonymized \
    --output_dir data/adversarial \
    --attack_types fgsm pgd deepfool \
    --epsilon 0.1

# Adversarial fine-tuning (5% injection ratio)
python scripts/train.py \
    --mode adv_finetune \
    --base_checkpoint checkpoints/dp_model \
    --adv_data_dir data/adversarial \
    --adv_ratio 0.05 \
    --output_dir checkpoints/final_model
```

### 4. IDS-LLM Validation

```bash
python src/validation/ids_llm.py \
    --input_dir outputs/generated_reports \
    --output_dir outputs/validated_reports \
    --threshold 0.1
```

### 5. Encrypted Inference

```bash
# Generate a physician key pair (run once per physician)
python src/encryption/encrypt_report.py --generate-keys \
    --physician_id dr_smith \
    --key_dir keys/

# Encrypt a validated report
python src/encryption/encrypt_report.py \
    --report outputs/validated_reports/report_001.txt \
    --physician_id dr_smith \
    --key_dir keys/ \
    --output outputs/encrypted/report_001.enc

# Decrypt (on physician's local device)
python src/encryption/decrypt_report.py \
    --encrypted outputs/encrypted/report_001.enc \
    --physician_id dr_smith \
    --key_dir keys/ \
    --output outputs/decrypted/report_001.txt
```

### 6. Running the Full Pipeline

```bash
# End-to-end: anonymize → generate → validate → encrypt
python scripts/inference.py \
    --image path/to/chest_xray.jpg \
    --notes "Patient notes here (optional)" \
    --checkpoint checkpoints/final_model \
    --physician_id dr_smith \
    --key_dir keys/ \
    --output_dir outputs/
```

### 7. Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/final_model \
    --test_data data/open-i/test \
    --metrics bleu ssim psnr mia phi_leakage \
    --output_dir results/
```

---

## 📈 Experimental Results

All results are obtained on the unmodified Open-I test set (1,541 pairs). Metrics are averaged over 3 independent runs.

### Med-Guard Ablation (PHI Anonymization)

| Configuration | BLEU-4 | PHI Leakage Rate |
|---|---|---|
| No Med-Guard (baseline) | 0.86 ± 0.01 | 18.7% |
| Text-only (Presidio) | 0.81 ± 0.02 | 6.3% |
| Image-only (σ=15) | 0.74 ± 0.02 | 14.9% |
| **Full Med-Guard** | **0.70 ± 0.01** | **2.1%** |

### Differential Privacy (DP-SGD) — Privacy–Utility Trade-off

| ε | MIA Accuracy | Utility (% of baseline) |
|---|---|---|
| 1.0 | 51.0% | 68.4% |
| 2.0 | 54.2% | 73.6% |
| **3.0** | **55.0%** | **81.1%** ← *Selected* |
| 4.0 | 59.4% | 83.0% |
| 5.0 | 64.5% | 85.6% |

Random-classifier MIA baseline = 50%. Unprotected baseline MIA = 85%.

### Adversarial Robustness (BLEU-4)

| Attack | Pre Fine-Tuning | Post Fine-Tuning | Improvement |
|---|---|---|---|
| FGSM (ε=0.1) | 0.45 | **0.68** | +51% |
| PGD (ε=0.1) | 0.33 | **0.63** | +91% |
| DeepFool | 0.29 | **0.51** | +76% |

### Prompt Injection Defense

| Training Strategy | Correct Response Rate |
|---|---|
| No defense | 37.5% |
| Adversarial fine-tuning | 67.3% |
| **Adv. fine-tuning + DP-SGD** | **78.3%** |

### IDS-LLM Validation Module

| Module | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Rule-Based | 95.1% | 92.3% | 93.7% | 0.93 |
| Clinical Parameters | 92.8% | 90.2% | 91.5% | 0.91 |
| Anomaly Detection (ML) | 90.4% | 87.6% | 89.0% | 0.90 |
| **Overall System** | **92.7%** | **90.1%** | **91.3%** | **0.94** |

### Security Component Ablation

| Configuration | BLEU-4 | MIA Acc. | Unsafe Output Rate |
|---|---|---|---|
| **Full system** | **0.70** | **55%** | **4.2%** |
| w/o DP-SGD | 0.74 | 78% | 5.1% |
| w/o Adversarial training | 0.73 | 61% | 12.8% |
| w/o IDS-LLM | 0.71 | 56% | 21.5% |

### Comparison with Baseline

| Method | BLEU-4 | MIA Acc. | PHI Leakage |
|---|---|---|---|
| MedViLL† (reference) | 0.82 | 85% | 18.7% |
| Presidio-only | 0.81 | 72% | 6.3% |
| DP-SGD Only | 0.75 | 60% | 15.2% |
| Adversarial Training Only | 0.78 | 70% | 16.1% |
| **SecureMed-LLM** | **0.70** | **55%** | **2.1%** |

†MedViLL trained on MIMIC-CXR + Open-I (95k+ pairs); not directly comparable due to different training data and no privacy protections.

### Inference Latency (CPU — Intel Xeon 2.0 GHz, 12 GB RAM)

| Component | Latency |
|---|---|
| Vision encoding (BioMedCLIP) | 0.8–2.5 s |
| Text generation (T5 decoder) | 2.5–5.5 s |
| IDS-LLM validation | 1.0–3.0 s |
| **End-to-end** | **6.5 ± 2.3 s/report** |

---

## 🔧 Configuration

Edit `configs/config.yaml` to control all hyperparameters:

```yaml
model:
  vision_encoder: "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
  text_decoder: "t5-base"
  image_resolution: 224
  max_seq_length: 128

training:
  optimizer: "AdamW"
  learning_rate: 2.0e-5
  batch_size: 16
  epochs: 5
  gradient_clip_norm: 1.0

differential_privacy:
  epsilon: 3.0
  delta: 1.0e-5
  noise_multiplier: 1.1
  max_grad_norm: 1.0
  accountant: "rdp"                    # Rényi DP accountant

anonymization:
  gaussian_sigma: 15                   # Med-Guard image noise level
  phi_entities:                        # Presidio entity types to redact
    - PERSON
    - DATE_TIME
    - LOCATION
    - MEDICAL_LICENSE

adversarial:
  injection_ratio: 0.05                # 5% adversarial samples per batch
  attack_types: ["fgsm", "pgd", "deepfool"]
  epsilon: 0.1                         # L-inf perturbation budget
  pgd_steps: 10

ids_llm:
  rule_file: "src/validation/rules.json"
  anomaly_contamination: 0.1
  embedding_model: "all-MiniLM-L6-v2"

encryption:
  curve: "curve25519"
  scheme: "ecies"

data:
  dataset_path: "data/open-i/"
  train_split: "train"
  val_split: "val"
  test_split: "test"
```

---

## ⚠️ Limitations

This study carries several important limitations that future work will address:

- **Single-dataset evaluation**: All experiments use the Open-I dataset from a single institution (Indiana University). Results should not be generalized to other modalities, languages, or patient populations.
- **Weak MIA attack**: A black-box confidence-thresholding MIA is used. Shadow model and likelihood ratio attacks may yield higher attack accuracy; the reported 55% MIA accuracy is an upper bound on privacy protection.
- **No white-box adversarial evaluation**: Robustness is assessed against transfer attacks from a ResNet18 surrogate, not direct white-box attacks on BioMedCLIP.
- **No clinical expert validation**: IDS-LLM rules were authored by the research team with reference to published guidelines, not validated by board-certified radiologists.
- **Encryption scope**: Only the final validated report is encrypted; intermediate embeddings and inference-time activations are not.
- **No formal regulatory compliance**: Components are technically consistent with HIPAA/GDPR PHI de-identification requirements, but formal certification has not been performed.

---

## 📖 Citation

If you use this code or build upon this work, please cite:

```bibtex
@article{boumezbeur2025securemed,
  title     = {SecureMed-LLM: A Privacy-Preserving Framework for Safeguarding Clinical Language Models},
  author    = {Boumezbeur, Aya and Harrag, Fouzi and Deriche, Mohamed and Khan, Muhammad},
  journal   = {PeerJ Computer Science},
  year      = {2025},
  note      = {Submitted for peer review, November 2025},
  institution = {University of Setif 1 Ferhat Abbas, Algeria;
                 Ajman University, UAE;
                 King Saud University, Saudi Arabia}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 📬 Contact

| Author | Affiliation | Contact |
|---|---|---|
| **Aya Boumezbeur** *(lead)* | LRSD Lab, University of Setif 1, Algeria | GitHub: [@boumezbeuraya](https://github.com/boumezbeuraya) |
| **Mohamed Deriche** *(corresponding)* | AI Research Centre, Ajman University, UAE | m.deriche@ajman.ac.ae |
| **Fouzi Harrag** | LRSD Lab, University of Setif 1, Algeria | — |
| **Muhammad Khan** | King Saud University, Saudi Arabia | — |

---

## 🙏 Acknowledgements

This work was supported by:
- Networks and Distributed Systems Lab (LRSD), Ferhat Abbas University — Project "Ethics of AI" (No. C00L07UN190120220005)
- Deanship of Research Graduate Studies, Ajman University — Projects 2024-IDG-ENIT-1 and 2025-IDG-ENIT-4

---

<div align="center">

**⭐ If you find this work useful, please consider starring the repository.**

</div>
