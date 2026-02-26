"""
Insurance Pricing Risk Modeling

Author: Michelle Regalado

Goal: Use regression models to predict annual medical insurance charges
and identify the main cost drivers (e.g., smoking, age, BMI).
This is structured as a real business case for an insurance company.
"""

# =========================================================
# 1. Imports
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# =========================================================
# 2. Data Loading & Cleaning
# =========================================================

filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'

# Load CSV without headers
df = pd.read_csv(filepath, header=None)
print("First 10 rows of raw data:")
print(df.head(10))

# Assign column names
headers = ["age", "gender", "bmi", "no_of_children", "smoker", "region", "charges"]
df.columns = headers

# Replace '?' markers with NaN
df.replace("?", np.nan, inplace=True)

print("\nInfo before cleaning:")
print(df.info())

# smoker is a categorical attribute, replaced with most frequent entry
is_smoker = df["smoker"].value_counts().idxmax()
df["smoker"].replace(np.nan, is_smoker, inplace=True)

# age is a continuous variable, replaced with mean age
mean_age = df["age"].astype("float").mean(axis=0)
df["age"].replace(np.nan, mean_age, inplace=True)

# Update data types
df[["age", "smoker"]] = df[["age", "smoker"]].astype("int")

print("\nInfo after cleaning:")
print(df.info())

# 2 decimals for charges
df[["charges"]] = np.round(df[["charges"]], 2)
print("\nCleaned data sample:")
print(df.head())


# =========================================================
# 3. Exploratory Data Analysis (EDA)
# =========================================================

# Regression plot: charges vs BMI
plt.figure(figsize=(6, 4))
sns.regplot(x="bmi", y="charges", data=df, line_kws={"color": "red"})
plt.ylim(0,)
plt.title("Charges vs BMI with Linear Fit")
plt.tight_layout()
plt.show()

# Boxplot: charges by smoker status
plt.figure(figsize=(6, 4))
sns.boxplot(x="smoker", y="charges", data=df)
plt.title("Charges by Smoking Status")
plt.tight_layout()
plt.show()

# Correlation matrix
print("\nCorrelation matrix:")
print(df.corr())


# =========================================================
# 4. Model Development
# =========================================================

# ----- Model 1: Linear regression with smoker only -----
lm = LinearRegression()
x = df[["smoker"]]
y = df[["charges"]]
lm.fit(x, y)
print("\nR² (smoker only):", lm.score(x, y))

# ----- Model 2: Linear regression with all variables -----
z = df[["age", "gender", "bmi", "no_of_children", "smoker", "region"]]
lm.fit(z, y)
print("R² (all features, linear):", lm.score(z, y))

# ----- Model 3: Polynomial regression with all variables -----
Input = [
    ("scale", StandardScaler()),
    ("polynomial", PolynomialFeatures(include_bias=False)),
    ("model", LinearRegression()),
]
pipe = Pipeline(Input)

z = z.astype(float)
pipe.fit(z, y)
ypipe = pipe.predict(z)
print("R² (polynomial, all features):", r2_score(y, ypipe))


# =========================================================
# 5. Model Refinement (Train/Test split + Ridge)
# =========================================================

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    z, y, test_size=0.2, random_state=1)

# ----- Ridge model (no polynomial) -----
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(x_train, y_train)
yhat = ridge_model.predict(x_test)
print("R² test (Ridge, linear):", r2_score(y_test, yhat))

# ----- Polynomial regression + Ridge model -----
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.transform(x_test)  # IMPORTANT: transform, not fit_transform

ridge_model.fit(x_train_pr, y_train)
y_hat = ridge_model.predict(x_test_pr)
print("R² test (Ridge, polynomial degree=2):", r2_score(y_test, y_hat))

    
 