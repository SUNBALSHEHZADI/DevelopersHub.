# DevelopersHub.

# Data Science Internship Projects

This repository contains the work completed during my Data Science Internship. The objective was to gain hands-on experience in building data-driven models across various domains such as HR analytics, summarization, healthcare, and finance.

## Table of Contents

1. [Task 1: Predict Employee Attrition](#task-1-predict-employee-attrition)
2. [Task 2: Text Summarization](#task-2-text-summarization)
3. [Task 3: Heart Disease Diagnosis Prediction](#task-3-heart-disease-diagnosis-prediction)
4. [Task 4: Loan Default Prediction](#task-4-loan-default-prediction)
5. [Technologies and Libraries](#technologies-and-libraries)
6. [How to Run the Code](#how-to-run-the-code)
7. [Conclusion](#conclusion)

## Task 1: Predict Employee Attrition

**Objective**: Build a classification model to predict whether an employee will leave a company based on HR data and derive actionable retention strategies.

**Dataset**: [IBM HR Analytics Dataset](https://www.kaggle.com/datasets/arnabchaki/ibm-hr-analytics)

### Steps:
- Performed **Exploratory Data Analysis (EDA)** to identify key factors influencing employee attrition.
- Trained classification models, such as **Random Forest** and **Logistic Regression**.
- Used **SHAP** for model interpretability to explain predictions.
- Derived actionable insights for HR retention strategies based on the model's findings.

### Outcome:
A classification model to predict employee attrition with actionable insights for HR retention strategies.

## Task 2: Text Summarization

**Objective**: Create a system that summarizes lengthy articles, blogs, or news into concise summaries.

**Dataset**: [CNN/Daily Mail Dataset](https://www.kaggle.com/datasets/cnn/daily-mail)

### Steps:
- Preprocessed the textual data for summarization tasks.
- Implemented **abstractive summarization** using pre-trained models like **BERT** via HuggingFace's **transformers** library.
- Fine-tuned the models for better summary quality and coherence.
- Evaluated the summary quality using real-world articles.

### Outcome:
A summarization model capable of generating concise summaries from long texts.

## Task 3: Heart Disease Diagnosis Prediction

**Objective**: Build a model to predict the likelihood of diseases such as heart disease based on medical data.

**Dataset**:  [Heart Disease Dataset](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)

### Steps:
- Conducted **Exploratory Data Analysis (EDA)** to understand the relationships between features and outcomes.
- Applied **feature selection** and **data scaling** to improve model performance.
- Trained various models such as **Gradient Boosting**, **SVM**, and **Neural Networks**.
- Evaluated model performance using metrics like **F1 Score**.
- Provided insights for healthcare professionals based on model predictions.

### Outcome:
A medical prediction model for early disease detection and prevention with actionable insights for healthcare professionals.

## Task 4: Loan Default Prediction

**Objective**: Build a classification model to predict whether a loan applicant will default using financial data.

**Dataset**: [Lending Club Loan Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

### Steps:
- Preprocessed the dataset by handling missing values and class imbalance using techniques like **SMOTE**.
- Trained classifiers on  **SVM**.
- Evaluated model performance using metrics like **Precision**, **Recall**, and **F1 Score**.
- Generated a comprehensive performance report and recommendations for lenders.

### Outcome:
A classification model to predict loan default, helping lenders reduce financial risks.

---

## Technologies and Libraries

- **Python**: For data manipulation, modeling, and visualization.
- **Pandas**: For data cleaning and preprocessing.
- **NumPy**: For numerical operations.
- **Matplotlib** & **Seaborn**: For data visualization.
- **scikit-learn**: For building and evaluating machine learning models.
- **XGBoost** : For boosting models.
- **HuggingFace Transformers**: For text summarization using pre-trained language models.
- **SHAP** & **LIME**: For model interpretability and explanation.

## How to Run the Code

1. Clone the repository:
   ```
   git clone https://github.com/SUNBALSHEHZADI/DevelopersHub.git
   
   ```
2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```
3. Navigate to the folder containing the project files for each task.
4. Run the Python scripts for each task:
   - For **Task 1**, run `employee_attrition_model.py`.
   - For **Task 2**, run `text_summarization.py`.
   - For **Task 3**, run `disease_diagnosis.py`.
   - For **Task 4**, run `loan_default_prediction.py`.

## Conclusion

During this internship, I worked on four tasks to develop and implement data-driven models in different domains: HR analytics, text summarization, disease prediction, and financial prediction. These projects helped me hone my skills in machine learning, data preprocessing, and model interpretability, and I gained valuable experience in working with diverse datasets and problem domains.
