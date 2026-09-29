# Automatic Identification of Brachycera (Diptera) Using Deep Learning

A computational pipeline for the automatic taxonomic identification of **Brachycera (Diptera)** specimens using standardized morphological images and convolutional neural networks (CNNs).

This project was developed in the context of an internship at the **Instituto de Higiene e Medicina Tropical (IHMT), Universidade NOVA de Lisboa**.

---

## Project Overview

Traditional taxonomic identification of Diptera is mainly based on morphological characteristics and requires specialized taxonomic knowledge. Deep learning and computer vision provide an alternative approach for supporting the automatic classification of biological specimens from images.

This project develops a reproducible computational workflow for the identification of Brachycera specimens from the IHMT entomological collection.

The workflow includes:

- dataset auditing;
- image and metadata validation;
- image preprocessing;
- preparation of supervised datasets;
- cross-validation;
- CNN training;
- model evaluation and comparison;
- analysis of different anatomical views;
- hierarchical taxonomic identification;
- evaluation of hierarchical predictions;
- generation of tables and visualisations.

---

## Objectives

The main objectives of the project are:

1. Obtain standardized photographs of specimens from the IHMT insect collection using predefined anatomical views.

2. Support the generation of new taxonomic classifications for previously unidentified specimens.

3. Apply and compare different CNN architectures for automatic taxonomic classification.

4. Evaluate the influence of different anatomical views on identification performance.

5. Develop an automated computational pipeline for metadata validation, image verification, preprocessing, dataset organization and model training.

6. Evaluate model performance at the **family** and **genus** taxonomic levels using quantitative classification metrics.

7. Apply hierarchical prediction so that genus-level predictions remain compatible with the predicted taxonomic family.

---

## Dataset

The dataset consists of morphological photographs of Brachycera specimens from the entomological collection of the Instituto de Higiene e Medicina Tropical.

Each specimen is associated with a unique collection identifier (`numCol`) and may contain multiple standardized anatomical views.

The four anatomical views considered in the project are:

| Code | Anatomical view |
|---|---|
| FDT | Dorsal view |
| FFF | Frontal view |
| FLP | Partial lateral view |
| FLT | Complete lateral view |

Taxonomic classification was investigated at two levels:

- **Family**
- **Genus**

Species-level classification was not used as the primary supervised classification target.

> **Data availability:** Raw specimen photographs are not included directly in this repository. The repository contains the computational workflow required for processing and analysing the dataset.

---

## Deep Learning Models

Three convolutional neural network architectures were evaluated:

- **EfficientNet-B0**
- **ResNet18**
- **MobileNetV3 Large**

The models were trained using transfer learning and evaluated across the four anatomical views.

Five-fold cross-validation was used to evaluate model performance.

---

## Evaluation Metrics

Model performance was evaluated using several classification metrics, including:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score

Confusion matrices and per-class metrics were also generated to analyse classification performance at family and genus levels.

---

## Hierarchical Identification

In addition to the independent CNN experiments, the project includes a hierarchical identification stage.

The hierarchical approach predicts the specimen's **family** and subsequently determines a compatible **genus**, preserving the biological relationship:

```text
Specimen
   ↓
Family
   ↓
Compatible Genus
```

This approach prevents taxonomically incompatible family–genus combinations and provides a structured method for assisting the identification of specimens.

---

## Computational Workflow

The complete analysis is automated using **Snakemake**.

The main workflow follows:

```text
Dataset Audit
      ↓
Image Validation
      ↓
Preprocessing
      ↓
Cross-validation
      ↓
CNN Training
      ↓
Result Aggregation
      ↓
Hierarchical Model Training
      ↓
Hierarchical Identification
      ↓
Hierarchical Evaluation
      ↓
Visualisation
```

Snakemake tracks dependencies between the different stages and executes only the steps that need to be generated or updated.

---

## Repository Structure

```text
snakemake_mosca/
│
├── Snakefile
├── README.md
│
├── config/
│
├── metadata/
│
├── scripts/
│   ├── plot_anatomy.py
│   ├── plot_architecture_comparison.py
│   ├── plot_architecture_boxplots.R
│   └── plot_confusion_matrices.py
│
├── src/
│   ├── core/
│   ├── identification/
│   ├── ml/
│   └── utils/
│
├── workflow/
│   └── envs/
│       ├── python_plot.yml
│       └── r_plot.yml
│
├── tests/
│
└── results/
    ├── evaluation/
    ├── figures/
    ├── plots/
    └── production/
```

---

## Reproducibility

The workflow uses Snakemake together with Conda environments to manage software dependencies.

The plotting environments are defined in:

```text
workflow/envs/python_plot.yml
workflow/envs/r_plot.yml
```

To execute the workflow:

```bash
snakemake --use-conda --cores 4
```

Snakemake automatically creates the required Conda environments when necessary.

To inspect what Snakemake will execute without actually running the workflow:

```bash
snakemake --use-conda -n -p
```

---

## Visualisations

The workflow automatically generates several visualisations used to evaluate the models, including:

- anatomical-view performance comparison;
- CNN architecture comparison;
- architecture performance boxplots;
- family confusion matrix;
- genus confusion matrix.

Python-based figures are generated using Matplotlib, while architecture performance boxplots are generated using R.

---

## Technologies

The project uses:

- Python
- PyTorch
- Torchvision
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Pillow
- R
- ggplot2
- Snakemake
- Conda

---

## Author

**Louis John Andrew Idehen**

Bioinformatics

Instituto Politécnico de Setúbal

Internship carried out at:

**Instituto de Higiene e Medicina Tropical (IHMT)**  
**Universidade NOVA de Lisboa**

---

## License

License information will be added before public release.
