import pandas as pd
import re
import gensim
from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pathlib import Path

# Load data
df = pd.read_csv("data/paragraphs.csv")

# Preprocessing tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_and_tokenize(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 3]

# Step 1: Tokenize and clean
token_lists = df["text"].astype(str).apply(clean_and_tokenize).tolist()

# Step 2: Build bigrams
bigram_model = Phrases(token_lists, min_count=3, threshold=10)
bigram_phraser = Phraser(bigram_model)
token_lists = [bigram_phraser[doc] for doc in token_lists]

# Step 3: Dictionary and corpus
dictionary = corpora.Dictionary(token_lists)
dictionary.filter_extremes(no_below=5, no_above=0.5)
corpus = [dictionary.doc2bow(text) for text in token_lists]

# Step 4: Train LDA
lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=5,
    random_state=2025,
    passes=10
)

# Step 5: Convert to topic distributions
def get_topic_vector(bow):
    topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)
    return [prob for _, prob in topic_dist]

topic_vectors = [get_topic_vector(bow) for bow in corpus]
topic_df = pd.DataFrame(topic_vectors, columns=[f"topic_{i}" for i in range(lda_model.num_topics)])

# Step 6: Add labels and export
topic_df["label"] = df["label"]
Path("data").mkdir(exist_ok=True)
topic_df.to_csv("data/lda_features.csv", index=False)

print("Saved topic distributions to data/lda_features.csv")
