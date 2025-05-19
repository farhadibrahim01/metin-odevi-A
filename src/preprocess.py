import re, pandas as pd, nltk, unidecode as ud
from pathlib import Path
nltk.download("stopwords", quiet=True)
STOPS = set(nltk.corpus.stopwords.words("english"))

def clean(t):
    t = ud.unidecode(t.lower())
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    t = " ".join(w for w in t.split() if w not in STOPS and len(w) > 2)
    return t

BASE = Path(__file__).resolve().parent.parent
df   = pd.read_csv(BASE / "data" / "processed" / "train_initial.csv")
df["text"] = df["text"].astype(str).map(clean)
df = df[df["text"].str.len() > 20]
df.to_csv(BASE / "data" / "processed" / "train_clean.csv", index=False)
print("train_clean.csv →", len(df), "rows")
