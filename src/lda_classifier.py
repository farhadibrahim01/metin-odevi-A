from pathlib import Path
import pandas as pd, numpy as np, joblib
from gensim import corpora, models
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report

BASE = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE / "data" / "processed" / "train_clean.csv")
df["topic"] = df["label"].str.extract(r"^(alcohol|dvi|paternity)")[0]

texts = [t.split() for t in df["text"]]

lda = models.LdaModel.load(str(BASE / "outputs" / "lda_model.gensim"))
dictionary = corpora.Dictionary.load(str(BASE / "outputs" / "lda_dictionary.dict"))
corpus = [dictionary.doc2bow(t) for t in texts]

def lda_vector(doc_bow, lda_model, num_topics):
    vec = [0.0] * num_topics
    for topic_id, prob in lda_model.get_document_topics(doc_bow):
        vec[topic_id] = prob
    return vec

X = np.array([lda_vector(bow, lda, lda.num_topics) for bow in corpus])
y = df["topic"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pred = cross_val_predict(LogisticRegression(max_iter=1000), X, y, cv=cv)

print("\n📊 LDA Tabanlı Logistic Regression Sonuçları\n")
print(classification_report(y, pred, zero_division=0))

model = LogisticRegression(max_iter=1000)
model.fit(X, y)
joblib.dump(model, BASE / "outputs" / "lda_logreg_model.joblib")
print("\n✅ lda_logreg_model.joblib kaydedildi.")
