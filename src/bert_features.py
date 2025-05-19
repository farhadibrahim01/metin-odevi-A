from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE / "data" / "processed" / "train_clean.csv")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

np.save(BASE / "outputs" / "bert_embeddings.npy", embeddings)
print("✅ BERT embeddings saved to outputs/bert_embeddings.npy")
