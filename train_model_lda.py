import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models

# Make sure these are downloaded once
# nltk.download('punkt')
# nltk.download('stopwords')

# Load data
df = pd.read_csv("data/paragraphs.csv")

# Preprocess text
stop_words = set(stopwords.words("english"))
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

df["tokens"] = df["text"].astype(str).apply(preprocess)

# Create dictionary and corpus
dictionary = corpora.Dictionary(df["tokens"])
corpus = [dictionary.doc2bow(text) for text in df["tokens"]]

# Train LDA model
lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=4, passes=10, random_state=2025)

# Assign dominant topic
def get_dominant_topic(doc_bow):
    topics = lda_model.get_document_topics(doc_bow)
    topics = sorted(topics, key=lambda x: x[1], reverse=True)
    return topics[0][0] if topics else -1

df["lda_topic"] = [get_dominant_topic(bow) for bow in corpus]

# Print topic keywords
for topic_id, topic_words in lda_model.print_topics(num_words=12):
    print(f"Topic {topic_id}: {topic_words}")

# Save to CSV
df.to_csv("data/paragraphs_with_lda_topics.csv", index=False)
