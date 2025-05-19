from pathlib import Path
from pdfminer.high_level import extract_text
import pandas as pd, re, itertools, unidecode as ud

BASE = Path(__file__).resolve().parent.parent
RAW  = BASE / "data" / "raw"
OUT  = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

def split_pdf_to_chunks(pdf_path, win=100, stride=60):
    txt = extract_text(pdf_path)
    words = txt.split()
    for i in range(0, len(words), stride):
        chunk = " ".join(words[i:i + win])
        if len(chunk.split()) >= 40:
            yield chunk

rows = []
for pdf in RAW.glob("*.pdf"):
    label = pdf.stem.lower().split()[0]
    if label == "forensic":
        label = "paternity"
    for chunk in split_pdf_to_chunks(pdf):
        rows.append({"text": chunk, "label": label})

pd.DataFrame(rows).to_csv(OUT / "train_initial.csv", index=False)
print("train_initial.csv →", len(rows), "rows")
