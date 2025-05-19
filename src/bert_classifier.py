from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import joblib

BASE = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE / "data" / "processed" / "train_clean.csv")
X = np.load(BASE / "outputs" / "bert_embeddings.npy")

df["topic"] = df["label"].str.extract(r"^(alcohol|dvi|paternity)")[0]

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear"
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(model, X, df["topic"], cv=cv)

print(classification_report(df["topic"], y_pred, zero_division=0))

model.fit(X, df["topic"])
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
joblib.dump(model, OUT / "bert_logreg_model.joblib")
