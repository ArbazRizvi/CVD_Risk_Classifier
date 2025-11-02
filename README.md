# Cardiovascular Disease Risk Classifier

## Objective

A key project from the Stanford/DeepLearning.AI specialization. The goal was to build and evaluate multiple machine learning models to predict the 10-year risk of cardiovascular disease (CVD) based on a patient's clinical data from the Framingham Heart Study.

## Methodology

1.  **Data Preprocessing:** Cleaned and preprocessed the dataset (`framingham.csv`). This included:
    * Handling missing values using median imputation.
    * One-hot encoding categorical features (e.g., `education`).
    * Scaling all features using `StandardScaler`.
2.  **Model Training:** Trained several classifiers using Scikit-learn:
    * Logistic Regression
    * K-Nearest Neighbors (KNN)
    * Support Vector Machine (SVM)
    * Random Forest
3.  **Evaluation:** Compared models based on Accuracy, Precision, Recall, and F1-Score. Given the class imbalance (fewer CVD cases), **F1-Score** was prioritized as the key metric.
4.  **Hyperparameter Tuning:** Used `GridSearchCV` to find the optimal parameters for the best-performing model (Random Forest) to maximize its F1-Score.

## Tech Stack

* Python
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Jupyter Notebook

## Result

The optimized Random Forest classifier achieved the **highest F1-score (89%)**, demonstrating a strong understanding of the end-to-end machine learning pipeline from data ingestion to model tuning and evaluation.

## How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/ArbazRizvi/CVD_Risk_Classifier.git](https://github.com/ArbazRizvi/CVD_Risk_Classifier.git)
    cd CVD_Risk_Classifier
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  Download the dataset from [Kaggle](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset) and place `framingham.csv` in the `data/` folder.
4.  Launch the Jupyter Notebook:
    ```bash
    jupyter notebook CVD_Risk_Analysis.ipynb
    ```
