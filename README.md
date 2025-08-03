<p align="left">
  <img src="https://aml.nik.uni-obuda.hu/themes/aml/assets/images/oe_nik_modern.png" style="width:1800px; height:190px; min-width:600px; min-height:100px; max-width:400px; max-height:64px;" />
</p>

# 🧠 Brain Tumor Classification  
### Multi-Stage Preprocessing & Machine Learning Analysis

---

<div align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/jupyter-notebook-orange?logo=jupyter">
  <img src="https://img.shields.io/badge/opencv-4.x-green?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/scikit--image-0.19%2B-yellow?logo=scikit-image" alt="scikit-image">
  <img src="https://img.shields.io/badge/scikit--learn-1.0%2B-blueviolet?logo=scikit-learn" alt="scikit-learn">
  <img src="https://img.shields.io/badge/pandas-1.3%2B-lightgrey?logo=pandas" alt="pandas">
  <img src="https://img.shields.io/badge/made%20with-NumPy-blue?logo=numpy" alt="NumPy">
  <img src="https://img.shields.io/badge/matplotlib-3.4%2B-yellowgreen?logo=matplotlib" alt="matplotlib">
  <img src="https://img.shields.io/badge/tqdm-supported-brightgreen?logo=python" alt="tqdm">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-blue?logo=windows" alt="Platform">
  <img src="https://img.shields.io/badge/Status-Research%20Project-informational" alt="Status">
  <img src="https://img.shields.io/badge/License-Restricted-red">
</div>

---

## 📝 Project Overview

A professional, modular pipeline for **brain tumor image classification** (3 tumor types) using advanced preprocessing, feature extraction, and dimensionality reduction.  
Supports multiple feature types (HOG, LBP, SIFT, GLCM, statistics) and preprocessing (CLAHE, Gaussian, Sobel, Canny).

Developed for **the BSc in Engineering Informatics, University of Óbuda in Hungary, AI specialization**.

<p>
  <strong>
    <span style="background-color:#ffeeba; color:#000000; padding:6px; border-radius:4px; display:inline-block;">⚠️ NOTE:</span>
  </strong>
  <i>Training and evaluation of the models were carried out using 
  <a href="https://lambdalabs.com/" target="_blank"><strong>Lambda GPU cloud servers</strong></a>, 
  as training locally would have been significantly slower and inefficient.</i>
</p>



---

## 📁 Folder Structure

```
.
├── preprocessing.ipynb
├── images/                  # Input images
├── output_images/           # Images sorted by class
├── image_classes.csv        # Image paths & classes
├── original_features/       # Extracted original features (CSV)
│   ├── canny/
│   │     ├── all_combined_canny_features.csv
│   │     ├── glcm_canny_features.csv
│   │     ├── hog_canny_features.csv
│   │     ├── hog_lbp_combined_canny_features.csv
│   │     ├── lbp_canny_features.csv
│   │     ├── sift_canny_features.csv
│   │     └── stat_canny_features.csv
│   ├── clahe/               # ⤷ has similar structure to `canny/`
│   ├── gaussian/            # ⤷ has similar structure to `canny/`
│   ├── original/            # ⤷ has similar structure to `canny/`
│   └── sobel/               # ⤷ has similar structure to `canny/`
│
├── feature_reduction/       # PCA-reduced features (CSV)
│   ├── canny/
│   │     ├── all_combined/
│   │     │        ├── all_combined_canny_pca_<n>.csv
│   │     │        └── all_combined_canny_variance_plot.csv
│   │     ├── glcm/
│   │     ├── hog/
│   │     ├── hog_lbp_combined/
│   │     ├── lbp/
│   │     ├── sift/
│   │     └── stat/
│   ├── clahe/               # ⤷ has similar substructure as `canny/`
│   ├── gaussian/            # ⤷ has similar substructure as `canny/`
│   ├── original/            # ⤷ has similar substructure as `canny/`
│   └── sobel/               # ⤷ has similar substructure as `canny/`
│
├── plots/                  # Diagrams by model reports(e.g., accuracy, F1 etc.)
│   ├── <model_name>_accuracy_boxplot.png
│   ├── <model_name>_f1_barplot.png
│   ├── <model_name>_weighted_f1_barplot.png
│   └── <model_name>_weighted_f1_feature_boxplot.png
│
├── models/                 # Trained models & reports
│   ├── decision_tree.ipynb
│   ├── decision_tree_results_with_reports.csv
│   ├── logistic_regression.ipynb
│   ├── logistic_regression_results_with_reports.csv
│   ├── nn.ipynb
│   ├── nn_results_with_reports.csv
│   ├── random_forest.ipynb
│   ├── random_forest_results_with_reports.csv
│   ├── svm.ipynb
│   ├── svm_results_with_reports.csv
│   ├── xgboost.ipynb
│   └── xgboost_results_with_reports.csv
│
└── README.md

```

---

## 🚀 Quick Start

1. **Clone the repository**
   ```sh
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install dependencies**
   ```sh
   pip install pandas opencv-python numpy scikit-image scipy scikit-learn matplotlib tqdm
   ```

3. **Prepare your dataset**
   - Place `.png` images in the `images/` folder, named as `id_class.png` (e.g., `123_1.png`).

---

## ⚙️ Usage

- Open `preprocessing.ipynb` in Jupyter or VS Code.
- Run all cells to:
  - Organize images & generate `image_classes.csv`
  - Preprocess images (resize, normalize, filter, enhance, edge detection)
  - Extract features (HOG, LBP, SIFT, GLCM, statistical)
  - Save original & PCA-reduced features as CSV

**Output:**
- `original_features/<preprocess_type>/<feature>_<preprocess_type>_features.csv`
- `feature_reduction/<preprocess_type>/<feature>_<preprocess_type>_pca_<n>.csv`
- Both `original_features/` and `feature_reduction/` folders contain variance curves (explained variance plots) for feature sets and PCA components.
- `models/` — Contains CSVs and metadata to help you select or identify the best model configuration (not actual model files)
- `plots/` — Contains model-wise comparisons of preprocessing techniques (e.g., barplots, boxplots) for each classifier

---

## 🖼️ Image Processing Pipeline

- **Preparation:** Resize and normalize images, denoise (Gaussian blur), enhance contrast (CLAHE).
- **Edge Detection:** Sobel and Canny algorithms.
- **Feature Extraction:**  
  - **HOG (Histogram of Oriented Gradients):** Captures edge and texture information.
  - **LBP (Local Binary Patterns):** Describes local texture.
  - **SIFT:** Robust keypoint-based features.
  - **GLCM (Gray-Level Co-occurrence Matrix):** Texture statistics.
  - **Statistical features:** Mean, standard deviation, skewness, entropy, Hu moments.
- **Dimensionality Reduction:** PCA (Principal Component Analysis).

---

## 📊 Results & Evaluation

The classification pipeline was evaluated using multiple machine learning models (Logistic Regression, SVM, Random Forest, XGBoost, Neural Network) and various preprocessing and feature extraction strategies. Performance was measured by accuracy and weighted F1-score using cross-validation.

**Key findings:**
- **Best results** were achieved with combined features (e.g., HOG+LBP or all_combined) and advanced preprocessing (CLAHE or Gaussian blur), especially after PCA-based dimensionality reduction.
- **XGBoost** and **Random Forest** consistently outperformed other models, reaching mean accuracy and weighted F1-scores above 0.93 with optimal feature sets.
- **Neural Networks** and **SVM** also performed well, with best accuracies around 0.91–0.92.
- **Edge-based preprocessing** (Sobel, Canny) alone was less effective, but improved results when combined with texture features.
- **Statistical features** alone yielded lower scores, but contributed positively in feature combinations.

**Typical metrics (best configurations):**
- *XGBoost*: Accuracy ≈ 0.94, Weighted F1 ≈ 0.94
- *Random Forest*: Accuracy ≈ 0.93, Weighted F1 ≈ 0.93
- *SVM / Neural Network*: Accuracy ≈ 0.91–0.92, Weighted F1 ≈ 0.91–0.92

See the `plots/` folder for detailed accuracy and F1-score distributions by preprocessing and feature type.

---

## 🤝 Contribution

<div style="background-color:#f8f9fa; border-left:6px solid red; padding:16px; border-radius:6px; font-size:15px; line-height:1.6;">
  <strong style="color:#000;">🚫 Currently not accepting unsolicited contributions.</strong><br>
  <i style="color:black;">For collaboration or suggestions, please contact the author directly.<br>
  All proposals will be considered on a case-by-case basis. </i>
</div>



---

## ⚠️ License

This project is **not openly licensed**.  
Any use, modification, or distribution requires explicit permission from the author.