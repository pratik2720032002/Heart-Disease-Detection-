
# Heart Disease Prediction Using Python and Supervised Machine Learning

**Author:** Pratik Kumar Sharma 
**Date:** May 2025  

---

## Project Overview
This repository implements five supervised machine-learning models to predict the presence of coronary artery disease (CAD) using the Cleveland Heart Disease dataset from the UCI Machine Learning Repository. The project covers:

- Data loading and cleaning  
- Label encoding and feature scaling  
- Model training, tuning, and evaluation  
- Comparative performance analysis  
- Explainability using feature importances and SHAP  

**Key Finding:** Logistic Regression achieved the highest test accuracy (92.6%) and AUC-ROC (0.93), with maximum heart rate, chest pain type, and ST depression as top predictors.

---

## Features
- **Data Preprocessing:** Handles missing values, label encoding, and z-score scaling  
- **Model Training:** Implements Logistic Regression, Decision Tree, Random Forest, Naive Bayes, and SVM  
- **Evaluation:** Computes accuracy, precision, recall, F1-score, specificity, and AUC-ROC; plots confusion matrices and ROC curves  
- **Explainability:** Extracts Random Forest impurity-based feature importances, Logistic Regression coefficients, and SHAP values  
- **Reproducibility:** All steps are documented in a Jupyter notebook; code is version-controlled on GitHub  

---

## Technologies Used
- **Language & Environment:** Python 3.8, Google Colab / Jupyter Notebook  
- **Libraries:** pandas, NumPy, scikit-learn, matplotlib, seaborn, shap  
- **Version Control:** Git & GitHub  

---

## Installation & Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/YourUsername/Heart-Disease-Prediction.git
   cd Heart-Disease-Prediction
   ```

2. **(Optional) Create and activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**  
   - Place `Heart_Disease_Prediction.csv` in the `data/` folder.

5. **Open the notebook**  
   ```bash
   jupyter notebook notebooks/Heart_Disease_Notebook.ipynb
   ```
   or upload to Google Colab and mount your Drive.

---

## Usage

1. **Preprocess the data** (in `preprocessing.py` or notebook cell)  
2. **Train and tune models** (in `modeling.py` or notebook cell)  
3. **Evaluate performance** (in `evaluation.py` or notebook cell)  
4. **Generate explainability plots** (in `explainability.py` or notebook cell)  

All results—tables and figures—are automatically saved to the `output/` directory.

---

## Project Structure
```
Heart-Disease-Prediction/
├── data/
│   └── Heart_Disease_Prediction.csv
├── notebooks/
│   └── Heart_Disease_Notebook.ipynb
├── src/
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── explainability.py
├── output/
│   ├── figures/
│   └── tables/
├── requirements.txt
├── README.md

```

---

## Results Summary
| Model               | Accuracy | AUC-ROC |
|---------------------|---------:|--------:|
| Logistic Regression |   0.926  |    0.93 |
| Naive Bayes         |   0.907  |    0.90 |
| SVM                 |   0.889  |    0.91 |
| Random Forest       |   0.796  |    0.84 |
| Decision Tree       |   0.704  |    0.78 |

---

## Explainability
- **Global Feature Importance:**  
  - Random Forest impurity scores and LR coefficients both highlight Max HR, Chest Pain Type, and ST Depression.  
- **Local Explanations:**  
  - SHAP summary plots show how high feature values (red) push predictions toward CAD presence and low values (blue) toward absence.

---

## References
1. Benjamin, E. J., Muntner, P., Alonso, A., Bittencourt, M. S., Callaway, C. W., Carson, A. P., … Virani, S. S. (2024). Heart disease and stroke statistics—2024 update: A report from the American Heart Association. *Circulation, 149*(3), e123–e147. https://doi.org/10.1161/CIR.0000000000001136
2. Krittanawong, C., Zhang, H., Wang, Z., Aydar, M., & Kitai, T. (2022). Artificial intelligence in precision cardiovascular medicine. *Journal of the American College of Cardiology, 80*(17), 1799–1810. https://doi.org/10.1016/j.jacc.2022.08.728  

--- 
