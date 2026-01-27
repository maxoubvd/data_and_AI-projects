# Customer Churn Prediction Analysis

## Project Overview

This project focuses on building a Machine Learning pipeline to predict customer churn (attrition). The goal is to identify customers who are likely to stop using the bank's services ("Exited") versus those who will stay ("Not exited").

The analysis places a strong emphasis on Recall over Precision, utilizing the F2-score as the primary performance metric. The business logic behind this decision is that identifying a potential churner (even if false positive) is more valuable than missing a churner entirely. 

## Team
- Youssef Benaddi
- Ali Abouachim-Alami
- Maxime Bendavid

## Key Features

- Data Preprocessing: Standardization of features using StandardScaler.
- Dimensionality Reduction: Implementation of Principal Component Analysis (PCA) to reduce feature space while retaining variance.
- Modeling: Implementation of classification models, including Logistic Regression (model_lr).
- Custom Evaluation: Optimization based on the F2-score to prioritize sensitivity (Recall).

## Technologies Used

- Python 3
- Data Manipulation: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Machine Learning: Scikit-learn (sklearn)
  - `PCA`
  - `StandardScaler`
  - `LogisticRegression`
  - `Decision Trees`
  - `Ensemble Models`
  - `Gradient Boosting Techniques`


## Methodology & Results

### The F2-Score Strategy

In customer churn prediction, False Negatives (predicting a customer will stay when they actually leave) are costly. Therefore, this project optimizes for the **F2-score**, which weighs Recall higher than Precision.

### Performance

As noted in the project conclusion, the final model adopts an aggressive strategy:

- **Recall**: ~75% (The model successfully identifies 3 out of 4 churning customers).
- **Precision**: ~50% (The model has a higher rate of false alarms).

While this results in a model that "over-predicts" churn (described in the analysis as slightly "psychopathic"), it ensures that the majority of at-risk clients are flagged for retention efforts.

## Visualizations

The notebook includes detailed visualizations, including:

- Confusion Matrices (visualizing True Positives vs. False Negatives).
- PCA component analysis.
- Data distribution plots using Seaborn.
- Comparisons of Models' performances.