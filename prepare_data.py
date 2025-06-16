import pandas as pd
import fitz  # PyMuPDF
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Step 1: PDF paths
pdf_map = {
    "alcohol": "data/Alcohol.pdf",
    "dvi": "data/DVI.pdf",
    "forensic": "data/Forensic Paternity.pdf"
}

# Step 2: Paragraph extraction
data = []
for label, path in pdf_map.items():
    doc = fitz.open(path)
    for page in doc:
        text = page.get_text().strip()
        paragraphs = re.split(r'\n\s*\n|(?<!\w)\n(?!\w)', text)
        for para in paragraphs:
            para = para.strip()
            if len(para.split()) > 10:
                data.append({"text": para, "source": label})
    doc.close()

df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 3: Assign semantic class labels
final_labels = []
for label in df["source"].unique():
    subset = df[df["source"] == label].copy()

    if label == "alcohol":
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        X = vectorizer.fit_transform(subset["text"])
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(X)
        subset["label"] = [f"alcohol_{i}" for i in clusters]
    else:
        subset["label"] = label

    final_labels.append(subset)

df_final = pd.concat(final_labels, ignore_index=True)

# Step 4: Save
Path("data").mkdir(exist_ok=True)
df_final.to_csv("data/paragraphs.csv", index=False)
print("Data saved to: data/paragraphs.csv")
