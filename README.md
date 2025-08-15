## SVM Binary Classification on Breast Cancer Wisconsin Dataset

📌 Overview

This project implements Support Vector Machine (SVM) for binary classification on the Breast Cancer Wisconsin (BCW) dataset. Both Linear and RBF kernels are used, and hyperparameters are tuned for improved performance. Decision boundaries are visualized for better understanding of model behavior.

--------

## 📂 Dataset  
We use the **Breast Cancer Wisconsin (Diagnostic) dataset** (`BCW.csv`), which contains 569 rows.  
For this project, only the following columns are used for training and visualization:

| Column Name      | Description |
|------------------|-------------|
| texture_mean     | Mean of gray-scale values |
| area_mean        | Mean area of the cell nuclei |
| diagnosis        | Target class (M = Malignant, B = Benign) |

---

📚 Libraries Used

- **pandas** → For dataset loading and preprocessing

- **numpy** → For numerical operations

- **matplotlib** → For decision boundary plotting

- **scikit-learn** → (SVC, Pipeline, GridSearchCV, cross_val_score, StandardScaler) → For SVM modeling, scaling, tuning, and evaluation

------------------

## ⚙️ Installation

- **pip install pandas numpy matplotlib scikit-learn**
  
-----------------

🛠 Changes Made

- **Removed unnecessary columns: id and Unnamed: 32**

- **Converted categorical target column diagnosis into numeric (M → 1, B → 0)**

- **Selected two features (radius_mean, texture_mean) for visualization**

- **Added decision boundary plots for both Linear and RBF SVM models**

- **Implemented GridSearchCV for hyperparameter tuning (C and gamma)**

- **Used cross-validation to evaluate performance**
