import pandas as pd
import numpy as np
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

base = "https://hastie.su.domains/ElemStatLearn/datasets/"
# spam.data: 57 features + last column is the response (0=non-spam, 1=spam)
# I used an AI tool to build the data pipeline as I could not find a simple way of importing it into Python.
spam = pd.read_csv(base + "spam.data", sep=r"\s+", header=None)
X = spam.iloc[:, :57]
y = spam.iloc[:, 57].astype(int).rename("is_spam")

# spam.traintest: 0=train, 1=test (same row order as spam.data)
tt = pd.read_csv(base + "spam.traintest", sep=r"\s+", header=None).iloc[:, 0].astype(int)

# Split 
X_train, y_train = X[tt == 0], y[tt == 0]
X_test, y_test = X[tt == 1], y[tt == 1]

print("Train/Test shapes:", X_train.shape, X_test.shape)
print("Train class balance:\n", y_train.value_counts().to_string())

# Parameters from assignment
Js = [2, 5, 10, 20, 50]

def run_gradient_boosting(X_train, y_train, X_test, y_test, Js):
    """
    Trains Gradient Boosting models for different tree complexities (J) and 
    plots the test deviance (log-loss) across boosting iterations (M).

   Args:
        X_train (pd.DataFrame): Training features for the Spam dataset.
        y_train (pd.Series): Binary training labels (0 for non-spam, 1 for spam).
        X_test (pd.DataFrame): Test features used for deviance evaluation.
        y_test (pd.Series): Binary test labels for evaluation.
        Js (list of int): List of terminal node values to test (e.g., [2, 5, 10, 20, 50]).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for J in Js:
        model = GradientBoostingClassifier(
            n_estimators=2500,
            learning_rate=0.05,
            max_leaf_nodes=J,
            max_depth=None,
            random_state=0,
        )
        model.fit(X_train, y_train)

        # Calculate Test Deviance
        test_deviance = np.zeros(model.n_estimators_, dtype=float)
        for i, proba in enumerate(model.staged_predict_proba(X_test)):
            test_deviance[i] = log_loss(y_test, proba)

        # Plot
        ax.plot(np.arange(1, model.n_estimators_ + 1), test_deviance, label=f"J={J}")
        
        # Print Best
        best_iter = np.argmin(test_deviance)
        print(f"J={J}: Min Deviance={test_deviance[best_iter]:.4f} at M={best_iter+1}")

    ax.set_xlabel("Number of Trees (M)")
    ax.set_ylabel("Test Deviance (Log Loss)")
    ax.set_title("Part B: Test Deviance vs Number of Trees")
    ax.legend(title="Terminal Nodes (J)")
    plt.tight_layout()
    plt.savefig("Test_Deviance.png", dpi=150)
    plt.show()

def run_CV_analysis(X_train, y_train, Js):
    """
    Performs 10-fold Stratified Cross-Validation for each value of J to find 
    the optimal number of trees (M) using the one-standard-error rule.

    Args:
        X_train (pd.DataFrame): Training features for the Spam dataset.
        y_train (pd.Series): Binary training labels for the Spam dataset.
        Js (list of int): List of terminal node values to evaluate via CV.
    """
    M_max = 1500
    skfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)

    for J in Js:
        cv_errors = np.zeros((10, M_max))
        base_model = GradientBoostingClassifier(
            n_estimators= M_max,
            learning_rate = 0.05,
            max_leaf_nodes= J,
            max_depth= None,
            random_state=0
        )

        # Iteratate over folds
        for fold_idx, (train_idx, val_idx) in enumerate(skfold.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            fold_model = clone(base_model)
            fold_model.fit(X_tr, y_tr)

            # Record error at every boosting stage
            for i, proba in enumerate(fold_model.staged_predict_proba(X_val)):
                cv_errors[fold_idx, i] = log_loss(y_val, proba)

        # Compute Mean and SE across folds
        cv_mean = cv_errors.mean(axis=0)
        cv_se = cv_errors.std(axis=0, ddof=1) / np.sqrt(cv_errors.shape[0])

        # 1-SE Rule
        min_idx = np.argmin(cv_mean)
        threshold = cv_mean[min_idx] + cv_se[min_idx]

        # First index where error <= treshold
        opt_M = np.where(cv_mean <= threshold)[0][0] +1
        print(f"J={J}: Optimal M (1-SE) = {opt_M}, CV Error = {cv_mean[opt_M-1]:.4f}")

        # Plot CV results
        fig, ax = plt.subplots(figsize=(8, 5))
        trees = np.arange(1, M_max + 1)
        ax.plot(trees, cv_mean, color='blue', label='CV Mean')
        ax.fill_between(trees, cv_mean - cv_se, cv_mean + cv_se, color='gray', alpha=0.2)
        
        ax.axvline(opt_M, color='red', linestyle='--', label=f'Optimal M={opt_M}')
        ax.axhline(threshold, color='green', linestyle=':', label='1-SE Threshold')
        
        ax.set_title(f"10-Fold CV (J={J})")
        ax.set_xlabel("Number of Trees")
        ax.set_ylabel("CV Deviance")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"CV_J_{J}.png", dpi=150)
        plt.show()
        
if __name__ == "__main__":
    run_gradient_boosting(X_train, y_train, X_test, y_test, Js)
    run_CV_analysis(X_train, y_train, Js)

from sklearn.metrics import accuracy_score

# Use the best J and M from CV results
best_J = 20 
best_M = 291 # optimal M

gbt_final = GradientBoostingClassifier(
    n_estimators=best_M,
    learning_rate=0.05,
    max_leaf_nodes=best_J,
    random_state=0
)
gbt_final.fit(X_train, y_train)
gbt_preds = gbt_final.predict(X_test)

print(f"GBT Test Accuracy: {accuracy_score(y_test, gbt_preds):.4f}")