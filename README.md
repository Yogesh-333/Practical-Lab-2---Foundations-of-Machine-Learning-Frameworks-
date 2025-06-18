# Practical Lab 2 — Multivariate Regression, Non-Parametric Models & Cross-Validation
  
Author: Yogesh Kumar Gopal  
Course: Foundations of Machine Learning Frameworks (CSCN8010)

---

##  Overview

This project involves building and evaluating multiple regression models to predict diabetes disease progression using the Scikit-learn diabetes dataset. The objective is to help physicians identify patients at risk using predictive modeling.

The models include:

- Univariate Polynomial Regression (on BMI)
- Multivariate Polynomial Regression
- Decision Trees
- k-Nearest Neighbors (kNN)

Model performance was evaluated using R², Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE), across Train, Validation, and Test sets.

---

##  Structure

The notebook is structured as follows:

### Part 1 — Data Preparation & EDA

- Load and understand the dataset
- Explore feature statistics and distributions
- Visualize relationships (scatter plots, histograms, correlation matrix)
- Data cleaning (if needed)
- Train/Validation/Test split (75% / 10% / 15%)

### Part 2 — Univariate Polynomial Regression

- Focus on BMI as the sole feature
- Fit Polynomial models from degree 0 to 5
- Evaluate R², MAE, MAPE on Train and Validation
- Select best degree based on performance
- Plot predictions and derive model equation
- Predict diabetes progression for a chosen BMI value
- Explain number of trainable parameters

Best univariate model: Polynomial Degree 5  
Equation: y = 154.97 (predicted for standardized BMI = -0.0001)  
Trainable parameters: 6 (intercept + 5 coefficients)

---

##  Results Summary (Part 3 — All Features)

| Model                   | R² Val     | MAE Val   | MAPE Val   | Trainable Parameters     |
|------------------------|------------|-----------|------------|--------------------------|
| Polynomial Degree 2    | 0.602991   | 37.75     | 41.10%     | 66                       |
| Polynomial Degree 3    | -115.9449  | 287.18    | 193.21%    | 286                      |
| Decision Tree Depth 3  | 0.412751   | 42.08     | 43.65%     | up to 22 nodes           |
| Decision Tree Depth 5  | 0.162955   | 51.55     | 51.77%     | up to 24 nodes           |
| kNN (k=3)              | 0.347063   | 50.77     | 49.45%     | non-parametric           |
| kNN (k=7)              | 0.429985   | 45.85     | 45.91%     | non-parametric           |

(Note: Values rounded to 2 decimal places for display)

---

##  Key Insights

- Polynomial Degree 2 provided the best generalization and lowest overall error.
- Polynomial Degree 3 clearly overfit the training data and failed on validation.
- Decision Trees performed decently at lower depths, but overfitting emerged with deeper trees.
- kNN models were stable but less accurate than polynomial regression.
- Including all features significantly improved prediction quality compared to using BMI alone.

---

##  Part 3 Conclusion

In this section, we evaluated multiple multivariate models using all available features from the diabetes dataset.

-  Polynomial Regression (Degree 2 and 3): Degree 2 achieved the most reliable results with minimal error and strong validation R². Degree 3 strongly overfit the training data, resulting in extreme errors and a negative R².
-  Decision Trees (Depth 3 and 5): Moderate performance with increasing risk of overfitting as depth increased. Depth 3 was more stable.
-  k-Nearest Neighbors (k=3 and k=7): k=7 outperformed k=3 with lower error and higher R², but both were less effective than Degree 2 polynomial regression.

 Overall, Polynomial Regression (Degree 2) achieved the best trade-off between accuracy and complexity. Its superior performance highlights the benefit of including all features and choosing a model that balances flexibility with generalization.

---

##  Requirements

- Python 3.12+
- Jupyter Notebook
- pandas, numpy, matplotlib, seaborn
- scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
