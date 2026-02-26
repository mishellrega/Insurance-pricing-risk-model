# Insurance Pricing Risk Modeling

## Project Overview

This project analyzes medical insurance data to model annual healthcare charges and identify the primary factors that influence cost variability.

The objective is to evaluate how demographic and behavioral attributes (such as age, BMI, and smoking status) impact insurance charges and to compare multiple regression approaches in terms of explanatory power and generalization performance.

---

## Dataset

The dataset includes the following variables:

- Age  
- Gender  
- BMI (Body Mass Index)  
- Number of children  
- Smoker status  
- Region  
- Annual medical insurance charges (target variable)

The target variable is continuous and represents annual healthcare expenditure per individual.

---

## Data Preparation

The following preprocessing steps were performed:

- Replaced missing values:
  - Mean imputation for continuous variables
  - Mode imputation for categorical variables
- Converted data types to numeric format
- Rounded the target variable (charges) to two decimal places
- Verified structure using `DataFrame.info()` and correlation analysis

---

## Exploratory Data Analysis

Key observations:

- Smoking status exhibits a strong positive relationship with insurance charges.
- BMI shows a moderate positive relationship with charges.
- Age contributes to cost variability, though to a lesser extent than smoking.
- The correlation matrix confirms smoking as the dominant predictor.

Visual analysis using regression plots and boxplots supports these findings.

---

## Modeling Approach

Four models were developed and compared:

### 1. Linear Regression (Smoker Only)
R² ≈ 0.62  
Smoking status alone explains approximately 62% of the variance in insurance charges.

### 2. Multiple Linear Regression (All Features)
R² ≈ 0.75  
Including demographic and health-related variables significantly improves model performance.

### 3. Polynomial Regression (Degree 2)
R² ≈ 0.85 (training data)  
Non-linear transformations capture interaction effects and improve explanatory power.

### 4. Polynomial + Ridge Regression
R² ≈ 0.78 (test data)  
Regularization improves generalization performance by reducing overfitting.

---

## Key Findings

- Insurance costs demonstrate non-linear relationships with predictor variables.
- Smoking status is the strongest individual driver of cost.
- Interaction effects (e.g., BMI × Smoker) improve predictive accuracy.
- Regularization is necessary to maintain performance on unseen data.

---

## Conclusion

The Polynomial Ridge Regression model provides the most balanced performance between explanatory power and generalization. This approach is suitable for risk-based pricing strategies and cost forecasting applications within an insurance context.

---

## Key Results

- Linear regression (smoker only): R² = 0.62  
- Linear regression (all variables): R² = 0.75  
- Polynomial regression: R² = 0.84  
- Ridge + Polynomial (test set): R² = 0.78  

---

## Business Interpretation

Smoking status is the strongest predictor of insurance charges.  
Incorporating additional demographic and behavioral variables improves model performance significantly.  

Non-linear interactions between age, BMI, and smoking further enhance predictive power.  
Regularization (Ridge) helps control overfitting and improves generalization on unseen data.

---

## Tools and Libraries

- Python  
- Pandas  
- NumPy  
- Seaborn  
- Matplotlib  
- Scikit-learn  

---

Author: Michelle Regalado