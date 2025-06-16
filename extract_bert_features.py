import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np

# Load cleaned 3-class data
df = pd.read_csv("data/paragraphs_3class.csv")

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to get CLS embedding for a paragraph
def get_cls_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_embedding

# Extract features
features = []
for text in tqdm(df["text"], desc="Extracting BERT embeddings"):
    emb = get_cls_embedding(str(text))
    features.append(emb)

# Convert to DataFrame and add labels
features_df = pd.DataFrame(features)
features_df["label"] = df["label"].values

# Save separately
features_df.to_csv("data/bert_features.csv", index=False)
print("BERT features saved to: data/bert_features.csv")
