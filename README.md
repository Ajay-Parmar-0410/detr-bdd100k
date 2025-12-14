[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/Ajay-Parmar-0410/detr-bdd100k/blob/main/notebooks/BDD100K_DETR_Object_Detection.ipynb
)


# DETR-based Object Detection on BDD100K (Sample)

This repository contains a simple **DETR (DEtection TRansformer)** pipeline for object detection on a **BDD100K sample dataset (COCO format)**.

✅ Includes:
- Training script (`src/train.py`)
- Inference script (`src/inference.py`)
- Colab notebook (`notebooks/BDD100K_DETR_Object_Detection.ipynb`) for easy reproduction

> Note: Dataset and trained model files are NOT included in this repo to keep it lightweight.

---

## Repository Structure

```bash
.
├── notebooks/
│   └── BDD100K_DETR_Object_Detection.ipynb
├── src/
│   ├── make_coco.py
│   ├── train.py
│   └── inference.py
├── .gitignore
└── README.md
