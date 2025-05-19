from pathlib import Path
import pandas as pd
import re
import gensim
from gensim import corpora, models
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords", quiet=True)

BASE = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE / "data" / "processed" / "train_clean.csv")
texts = df["text"].astype(str).tolist()

# Tokenization & stop word removal
STOPWORDS = set(stopwords.words("english"))
docs = [[w for w in re.findall(r"\b[a-z]{3,}\b", t.lower()) if w not in STOPWORDS]
        for t in texts]

# Dictionary & Corpus
dictionary = corpora.Dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]

# LDA model
lda = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10, random_state=42)

# Save model & data
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
lda.save(str(OUT / "lda_model.gensim"))
dictionary.save(str(OUT / "lda_dictionary.dict"))
corpora.MmCorpus.serialize(str(OUT / "lda_corpus.mm"), corpus)

# Show topics
for i, topic in lda.print_topics(num_words=10):
    print(f"Topic {i+1}: {topic}")

print("\n✅ Saved: LDA model, dictionary, and corpus")
