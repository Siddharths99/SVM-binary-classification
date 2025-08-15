import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#LoadING and preparing dataset for binary classification
df = pd.read_csv("BCW.csv")
df = df.drop(columns=["id", "Unnamed: 32"])
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

FEATURES = ["radius_mean", "texture_mean"]
TARGET_COL = "diagnosis"

X = df[FEATURES].values
y = df[TARGET_COL].values

#Train an SVM with linear and RBF kernel.
linear_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="linear", random_state=42))
]).fit(X, y)

rbf_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", random_state=42))
]).fit(X, y)

#Visualize decision boundary using 2D data.
def plot_decision_boundary(ax, X2, y2, clf, title):
    x_min, x_max = X2[:,0].min()-0.5, X2[:,0].max()+0.5
    y_min, y_max = X2[:,1].min()-0.5, X2[:,1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap=plt.cm.coolwarm)
    ax.scatter(X2[:,0], X2[:,1], c=y2, edgecolor="k", cmap=plt.cm.coolwarm)
    ax.set_xlabel(FEATURES[0]); ax.set_ylabel(FEATURES[1]); ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(12,5))
plot_decision_boundary(axes[0], X, y, linear_clf, "SVM Linear")
plot_decision_boundary(axes[1], X, y, rbf_clf, "SVM RBF")
plt.tight_layout()
plt.show()

#Tune hyperparameters like C and gamma.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 1, 0.1, 0.01, 0.001]
}
rbf_grid = GridSearchCV(
    Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf"))]),
    param_grid, cv=cv, n_jobs=-1
).fit(X, y)

print("Best params (RBF):", rbf_grid.best_params_)
print("Best CV accuracy (RBF):", round(rbf_grid.best_score_, 4))

#Evaluating Performance usung cross validation
lin_scores = cross_val_score(linear_clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
rbf_scores = cross_val_score(rbf_grid.best_estimator_, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

print(f"Linear SVM CV Accuracy: {lin_scores.mean():.3f} ± {lin_scores.std():.3f}")
print(f"RBF (tuned) SVM CV Accuracy: {rbf_scores.mean():.3f} ± {rbf_scores.std():.3f}")