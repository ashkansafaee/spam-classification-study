# Spam Classification — Gradient Boosting vs SVM

A comparative study of two ensemble/kernel methods on the classic Hastie et al. spam dataset. Built as part of coursework in Statistical Learning at Stockholm University.

## Problem

Binary classification: given 57 email features (word frequencies, character frequencies, run-length statistics), predict whether an email is spam or not. The goal is not just accuracy — it is understanding *how* each model arrives at its answer and how to tune it rigorously.

## Models

### Gradient Boosting (Project 2)
- Trained GBT classifiers across 5 tree complexities: J ∈ {2, 5, 10, 20, 50} terminal nodes
- Tracked test deviance (log-loss) across up to 2,500 boosting iterations to identify overfitting behaviour
- Applied 10-fold stratified cross-validation with the **one-standard-error rule** to select the optimal number of trees M without overfitting to the validation set
- Best configuration: J=20, M=291 → **Test accuracy: ~95%**

### Support Vector Machine (Project 3)
- Exploratory analysis revealed heavy right-skew in several features; log-transformation applied
- GridSearchCV across 4 kernels (linear, RBF, polynomial, sigmoid) with full hyperparameter sweep
- Best kernel: RBF with C=10, γ=0.01
- 10-fold CV with 1-SE rule applied to select the simplest model within one standard error of the minimum
- Confusion matrix analysis to understand precision/recall trade-offs
- **Test accuracy: ~94%**

## Key Findings

- Gradient Boosting with moderate tree complexity (J=20) outperformed deeper trees, confirming that boosting benefits from weak learners
- The 1-SE rule consistently selected simpler models than pure minimum-CV, with negligible loss in accuracy — a practically important finding for deployment
- RBF-SVM and GBT reached comparable accuracy (~94–95%), but GBT was more interpretable via feature importance and staged prediction diagnostics

## Stack

Python · Scikit-learn · NumPy · Pandas · Matplotlib · Seaborn

## Data

Hastie, Tibshirani & Friedman spam dataset — 4,601 emails, 57 features, pre-defined train/test split.
Source: `https://hastie.su.domains/ElemStatLearn/datasets/`

---

*Part of a B.Sc. in Mathematical Statistics and Machine Learning, Stockholm University.*
