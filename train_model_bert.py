import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load BERT features
df = pd.read_csv("data/bert_features.csv")
X = df.drop(columns=["label"])
y = df["label"]

# PCA to reduce dimensions
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.4, stratify=y, random_state=2025
)

# Model
model = LogisticRegression(max_iter=1000, class_weight="balanced", solver='saga')

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-Validation Scores: {np.round(cv_scores, 4)}")
print(f"Average CV Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean() * 100:.2f}%)")

# Train and evaluate
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)
labels = sorted(y.unique())

# Output
print("\nConfusion Matrix:")
print(pd.DataFrame(cm, index=labels, columns=labels))
print(f"\nAccuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

