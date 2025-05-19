from pathlib import Path
import pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction import text

BASE = Path(__file__).resolve().parent.parent
df   = pd.read_csv(BASE / "data" / "processed" / "train_clean.csv")
df["topic"] = df["label"].str.extract(r"^(alcohol|dvi|paternity)")[0]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=list(text.ENGLISH_STOP_WORDS),
        max_features=2000,
        min_df=2,
        sublinear_tf=True)),
    ("clf", LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced"))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(pipe, df["text"], df["topic"], cv=cv)

print(classification_report(df["topic"], y_pred, zero_division=0))

pipe.fit(df["text"], df["topic"])
OUT = BASE / "outputs"; OUT.mkdir(exist_ok=True)
joblib.dump(pipe, OUT / "model_tfidf_logreg_balanced.joblib")
print("model_tfidf_logreg_balanced.joblib saved")
