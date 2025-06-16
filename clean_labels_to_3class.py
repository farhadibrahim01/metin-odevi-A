import pandas as pd
import os

# Load original file
df = pd.read_csv("data/paragraphs.csv")

# Merge alcohol_0 and alcohol_1 into a single "alcohol" label
df["label"] = df["label"].replace({
    "alcohol_0": "alcohol",
    "alcohol_1": "alcohol"
})

# Save to new file
os.makedirs("data", exist_ok=True)
df.to_csv("data/paragraphs_3class.csv", index=False)

print("Clean 3-class version saved to: data/paragraphs_3class.csv")
