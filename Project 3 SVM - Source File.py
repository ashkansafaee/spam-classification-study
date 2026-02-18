# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

#### DATA LOADING AND PREPROCESSING
# Data Loading
base = "https://hastie.su.domains/ElemStatLearn/datasets/"
spam = pd.read_csv(base + "spam.data", sep=r"\s+", header=None)
X = spam.iloc[:, :57]
y = spam.iloc[:, 57].astype(int).rename("is_spam")

tt = pd.read_csv(base + "spam.traintest", sep=r"\s+", header=None).iloc[:, 0].astype(int)

# Split 
X_train, y_train = X[tt == 0], y[tt == 0]
X_test, y_test = X[tt == 1], y[tt == 1]

# Scaling
# I used ChatGPT to explore the best way to perform the scaling and it suggested sklearn's 
# preprocessing tool StandardScaler which was implemented
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train.describe()

# %%
#### Exploratory data analysis visualizations

# Identify skewed features in training data
high_value_features = X_train.columns[X_train.max() > 100]
print("Skewness of high-value features:\n", X_train[high_value_features].skew())

# Visualize Feature 55
plt.figure(figsize=(14, 5))

# Plot 1: Raw Data
plt.subplot(1, 2, 1)

# Using .values ensures Seaborn doesn't get confused by indices
sns.histplot(X_train.iloc[:, 55].values, bins=50, kde=True, color='salmon')
plt.title(f"Raw Feature 55 (Max: {X_train.iloc[:, 55].max()})")
plt.xlabel("Sequence Length")

# Plot 2: Log-Transformed
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(X_train.iloc[:, 55].values), bins=50, kde=True, color='skyblue')
plt.title("Log-Transformed Feature 55")
plt.xlabel("log(1 + Sequence Length)")

plt.tight_layout()
plt.show()


# %%
#### GRIDSEARCHCV OVER DIFFERENT KERNELS
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Using inspiration from https://www.kaggle.com/code/prashant111/svm-classifier-tutorial#8.-Exploratory-data-analysis- 
# I found that performing a GridSearchCV with different parameters 
# would be the best approach to evaluate hyperparameters for different SVC kernels. 

param_grid = [
    # Linear Kernel : Only needs C
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},

    # RBF Kernel: Needs C and Gamme
    {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1]},

    # Polnomial Kernal : Needs C and degree of polynomial
    {'kernel': ['poly'], 'C':[1, 100], 'degree':[2, 3]},

    # Sigmoid Kernel : Needs C
    {'kernel': ['sigmoid'], 'C': [1, 10]}
]

grid_search = GridSearchCV(SVC(cache_size=1000), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train_scaled, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

final_model = grid_search.best_estimator_
test_acc = final_model.score(X_test_scaled, y_test)
print(f"Final Test Accuracy: {test_acc:.4f}")

# %%
from sklearn.metrics import accuracy_score
#### IMPLEMENT MODEL WITH OPTIMAL HYPERPARAMETERS
# Implementing the model with optimal parameters from GridSearchCV
svc = SVC(kernel='rbf', C = 10, gamma=0.01)
svc.fit(X_train_scaled, y_train)
y_pred = svc.predict(X_test_scaled)

# Print model accuracy with RBF Kernel and optimal hyperparameters
print(f'Model accuracy with RBF kernel and C={10}, Gamma={0.01}: {accuracy_score(y_test, y_pred):.4f}')
print('Training set score RBF kernel: {:.4f}'.format(svc.score(X_train_scaled, y_train)))
print('Test set score RBF kernel: {:.4f}'.format(svc.score(X_test_scaled, y_test)))

# %%
#### NULL ACCURACY
# Calculating null accuracy to compare with training and test accuracy
y_test.value_counts()
null_accuracy = (941/(941+595))
print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

# %%
#### CONFUSION MATRIX
from sklearn.metrics import confusion_matrix

# Implement confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])

# %%
# Create heatmap for confusion matrix
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

# %%
#### CROSS-VALIDATION PLOT 
from sklearn.model_selection import cross_validate, KFold, cross_val_score
import numpy as np

# Prepare to loop over different C's and save errors and std errors for plot
C_range = np.logspace(-1, 2, 10)
cv_errors = []
std_errors = []

# Kfold with 10 splits 
kf = KFold(n_splits=10, shuffle=True, random_state=0)

# Loop over different C's with optimal hyperparameters from GridSearchCV
for c in C_range:
    svc = SVC(kernel='rbf', C=c, gamma = 0.01)
    errors = 1 - cross_val_score(svc, X_train_scaled, y_train, cv = kf)

    # Compute Mean and SE across folds
    cv_errors.append(np.mean(errors))
    std_errors.append(np.std(errors) / np.sqrt(10))

# Identify the minimum error and the 1-SE threshold
min_idx = np.argmin(cv_errors)
min_error = cv_errors[min_idx]
se_at_min = std_errors[min_idx]
threshold = min_error + se_at_min

# Find the simplest model (smallest C) within the threshold
optimal_c_idx = np.where(cv_errors <= threshold)[0][0]
optimal_c = C_range[optimal_c_idx]

# Visualization
plt.figure(figsize=(10, 6))
plt.errorbar(np.log10(C_range), cv_errors, yerr=std_errors, fmt='-o', capsize=5, label='CV Error')
plt.axhline(y=threshold, color='r', linestyle='--', label='1-SE Threshold')
plt.axvline(x=np.log10(C_range[min_idx]), color='green', linestyle=':', label='Min Error C')
plt.axvline(x=np.log10(optimal_c), color='purple', linestyle='-', label='Optimal C (1-SE Rule)')

plt.xlabel('log10(C)')
plt.ylabel('10-Fold CV Error Rate')
plt.title('CV Error vs. Model Complexity (One-Standard-Error Rule)')
plt.legend()
plt.show()

print(f"Minimum Error at C={C_range[min_idx]:.2f}")
print(f"Optimal Model (1-SE Rule) at C={optimal_c:.2f}")


