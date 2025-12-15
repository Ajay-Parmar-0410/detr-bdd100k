[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/Ajay-Parmar-0410/detr-bdd100k/blob/main/notebooks/BDD100K_DETR_Object_Detection.ipynb)

# DETR-based Object Detection on BDD100K (Sample)

This repository demonstrates an **end-to-end object detection pipeline using DETR (DEtection TRansformer)** on a **sample subset of the BDD100K dataset**, formatted in **COCO style**.

The project is designed to be **fully reproducible on Google Colab**, allowing reviewers to run the complete pipeline without any local setup.

---

## ✅ Key Features

- DETR with **ResNet-50 backbone**
- COCO-format annotations
- Lightweight **BDD100K sample dataset**
- End-to-end pipeline:
  - Dataset download & extraction from Google Drive
  - Annotation verification
  - Sanity training
  - Inference & visualization

> ⚠️ Dataset files and trained weights are **not stored in this repository** to comply with GitHub size limits.  
> The dataset is automatically downloaded from **Google Drive** inside the Colab notebook.

---

## 📂 Repository Structure

```bash
.
├── notebooks/
│   └── BDD100K_DETR_Object_Detection.ipynb
├── src/
│   ├── make_coco.py        # COCO conversion utility
│   ├── train.py            # DETR training script
│   └── inference.py        # Inference & visualization
├── .gitignore
└── README.md
