import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Load LDA topic features
df = pd.read_csv("data/lda_features.csv")
X = df[[col for col in df.columns if col.startswith("topic_")]]
y = df["label"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=2025
)

# Train classifier
clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=2025
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Cross-validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Cross-Validation Scores:", np.round(cv_scores, 4))
print(f"Average CV Accuracy: {np.mean(cv_scores):.4f} ({np.mean(cv_scores) * 100:.2f}%)\n")

# Evaluation metrics
cm = confusion_matrix(y_test, y_pred)
label_names = sorted(list(set(y_test)))

print("Confusion Matrix (labeled):")
print(pd.DataFrame(cm, index=[f"True: {l}" for l in label_names],
                      columns=[f"Pred: {l}" for l in label_names]))
print()

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print(f"Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

