[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/Ajay-Parmar-0410/detr-bdd100k/blob/main/notebooks/BDD100K_DETR_Object_Detection.ipynb
)

# DETR-based Object Detection on BDD100K (Sample)

This repository demonstrates an **end-to-end object detection pipeline using DETR (DEtection TRansformer)** on a **sample subset of the BDD100K dataset**, formatted in **COCO style**.

The project is designed to be **fully reproducible on Google Colab**, enabling reviewers to run the complete pipeline without local setup.

---

## ✅ Key Features
- DETR with **ResNet-50 backbone**
- COCO-format annotations
- Lightweight **BDD100K sample dataset**
- End-to-end pipeline:
  - Dataset download & extraction
  - Annotation verification
  - Sanity training
  - Inference & visualization

> ⚠️ Dataset files and trained weights are **not stored in this repository** to keep it lightweight.  
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
---

## Reproducibility (For Reviewers)

This project is fully reproducible using Google Colab.

### Steps:
1. Click the **Open in Colab** badge above
2. Run all cells sequentially
3. Approve Google Drive access when prompted
4. The notebook will:
   - Download a BDD100K sample dataset from Google Drive
   - Verify COCO annotations
   - Run a short sanity training
   - Perform inference and visualize predictions

> Note: Full BDD100K is not included due to size constraints.  
> A representative sample is used for demonstration.
