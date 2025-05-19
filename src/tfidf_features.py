from pathlib import Path
import pandas as pd, numpy as np, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

BASE = Path(__file__).resolve().parent.parent
df   = pd.read_csv(BASE / "data" / "processed" / "train_clean.csv")
OUT  = BASE / "outputs"
OUT.mkdir(exist_ok=True)

vec = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words=list(text.ENGLISH_STOP_WORDS),
    max_features=2000,
    sublinear_tf=True,
    min_df=2
)
X = vec.fit_transform(df["text"])
vocab = np.array(vec.get_feature_names_out())

label_terms = {}
for lab in df["label"].unique():
    mask = (df["label"] == lab).values
    scores = X[mask].mean(axis=0).A1
    label_terms[lab] = vocab[scores.argsort()[::-1][:15]].tolist()

pd.Series(label_terms).to_csv(OUT / "tfidf_top_terms.csv")
joblib.dump(vec, OUT / "tfidf_vectorizer.joblib")

print("tfidf_top_terms.csv and tfidf_vectorizer.joblib saved")
