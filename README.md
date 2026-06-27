# CreditWise

CreditWise is a machine learning project designed to predict loan approval decisions based on applicant financial and demographic data. It leverages multiple classification algorithms and feature engineering techniques to improve predictive performance and support data-driven decision-making.

---

## Project Overview

Financial institutions process a large number of loan applications daily and require efficient, reliable evaluation systems.  
CreditWise addresses this problem by building predictive models trained on historical loan data, using key indicators such as credit score, income, and debt-to-income ratio to determine loan approval outcomes.

---

## Dataset

The dataset (`loan_approval_data.csv`) contains a mix of financial and demographic features, including:

- Applicant Income  
- Co-applicant Income  
- Credit Score  
- Debt-to-Income (DTI) Ratio  
- Loan Amount  
- Savings  
- Age  
- Loan Purpose  
- Employment Status  
- Employer Category  
- Property Area  
- Gender  
- Marital Status  
- Education Level  

**Target Variable:** Loan Approval Status  

---

## Methodology

### Data Preprocessing
- Handled missing values using **SimpleImputer**
  - Mean strategy for numerical features  
  - Most frequent strategy for categorical features  
- Removed `Applicant_ID` as it does not contribute to prediction  

### Exploratory Data Analysis (EDA)
- Visualized distributions of loan purpose, employment status, and income  
- Used box plots to compare approved vs rejected loans across:
  - Credit Score, DTI Ratio, Income, Savings, Age, Loan Amount  
- Generated a correlation heatmap to analyze feature relationships

### 🔍 Insights from EDA & Correlation Analysis

- **Credit Score** shows a strong positive correlation with loan approval, making it one of the most influential features  
- **Debt-to-Income (DTI) Ratio** is negatively correlated with approval, indicating higher debt reduces approval chances

### Feature Encoding
- Applied **Label Encoding** for:
  - Education Level  
  - Loan Approval Status  
- Applied **One-Hot Encoding** for remaining categorical variables  

### Model Training
- Split dataset into **80% training** and **20% testing**  
- Applied **StandardScaler** for feature normalization  

Trained and evaluated the following models:
- K-Nearest Neighbors (k = 13)  
- Logistic Regression  
- Gaussian Naive Bayes  

### Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  

### Feature Engineering
- Introduced polynomial features:
  - `DTI_Ratio²`  
  - `Credit_Score²`  
- Retrained all models to assess performance improvements  

---
## 📊 Results

### K-Nearest Neighbors (k = 13)

| Metric       | Before Feature Engineering | After Feature Engineering |
|--------------|---------------------------|---------------------------|
| Accuracy     | 79.0%                     | 78.5%                     |
| Precision    | 73.17%                    | 73.68%                    |
| Recall       | 49.18%                    | 45.90%                    |
| F1 Score     | 58.82%                    | 56.57%                    |

**Insight:** Feature engineering did not significantly improve KNN. Slight gain in precision but drop in recall and F1-score indicates weaker detection of approved loans.

---

### Logistic Regression

| Metric       | Before Feature Engineering | After Feature Engineering |
|--------------|---------------------------|---------------------------|
| Accuracy     | 86.5%                     | 87.5%                     |
| Precision    | 78.33%                    | 79.03%                    |
| Recall       | 77.05%                    | 80.33%                    |
| F1 Score     | 77.69%                    | 79.67%                    |

**Insight:** Best performing model. Feature engineering improved all metrics, especially recall and F1-score, showing strong balance and generalization.

---

### Gaussian Naive Bayes

| Metric        | Before Feature Engineering | After Feature Engineering |
|--------------|---------------------------|---------------------------|
| Accuracy     | 86.5%                     | 86.5%                     |
| Precision    | 80.36%                    | 78.33%                    |
| Recall       | 73.77%                    | 77.05%                    |
| F1 Score     | 76.92%                    | 77.69%                    |

**Insight:** Stable performance. Feature engineering improved recall and F1-score but slightly reduced precision.

---

## 🏆 Final Model Comparison after Feature Engineering

| Model               | Accuracy  | Precision | Recall | F1 Score |
|---------------------|-----------|-----------|--------|----------|
| Logistic Regression | 87.5%     | 79.03%    | 80.33% | 79.67%   |
| Naive Bayes         | 86.5%     | 78.33%    | 77.05% | 77.69%   |
| KNN (k=13)          | 78.5%     | 73.68%    | 45.90% | 56.57%   |

---

## 🔍 Key Insights

- Logistic Regression achieved the best overall performance  
- Feature engineering significantly improved linear model performance  
- KNN struggled with recall, likely due to class imbalance  
- Naive Bayes showed stable but moderate performance  
- Credit Score and DTI Ratio are strong predictors of loan approval
   
---

## 📈 Conclusion

Feature engineering had a positive impact on model performance, particularly for Logistic Regression, which achieved the highest accuracy and balanced metrics.  
This project demonstrates the importance of model selection and feature transformation in building reliable financial prediction systems.

---

## Tech Stack

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

---

## How to Run

1. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
