import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Step 1: Load dataset
df = pd.read_csv("data/paragraphs.csv")
X = df["text"]
y = df["label"]

# Step 2: Define pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 3),  # Includes unigrams, bigrams, trigrams
        min_df=2,
        max_df=0.9,
        stop_words="english"
    )),
    ("clf", LinearSVC(
        class_weight="balanced",
        max_iter=2000,
        random_state=202
    ))
])

# Step 3: Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)
cv_scores = cross_val_score(pipeline, X, y, cv=skf)
print("Cross-Validation Scores:", cv_scores)
print(f"Average CV Accuracy: {np.mean(cv_scores):.4f} ({np.mean(cv_scores) * 100:.2f}%)\n")

# Step 4: Final train/test split evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=2025
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Step 5: Evaluation metrics
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print(f"Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
