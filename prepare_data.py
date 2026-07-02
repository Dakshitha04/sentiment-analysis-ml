import pandas as pd

# Column names
cols = ["sentiment", "id", "date", "query", "user", "text"]

# Load dataset
df = pd.read_csv(
    "data/training.1600000.processed.noemoticon.csv",
    names=cols,
    encoding="latin-1"
)

# Keep required columns
df = df[["text", "sentiment"]]

# 🔥 REMOVE ALL PROBLEM DATA
df = df.dropna()  # remove NaN
df["text"] = df["text"].astype(str)  # ensure string
df = df[df["text"].str.strip() != ""]  # remove empty text

# Convert labels
df["sentiment"] = df["sentiment"].map({
    0: "negative",
    4: "positive"
})

# Remove any unmapped values
df = df.dropna()

# Add neutral data
neutral_data = pd.DataFrame({
    "text": ["It is okay", "Average product", "Nothing special", "Fine", "Normal"] * 200,
    "sentiment": ["neutral"] * 1000
})

# Combine dataset
df = pd.concat([df.sample(50000), neutral_data], ignore_index=True)

# Save clean dataset
df.to_csv("data/dataset.csv", index=False)

print("Clean dataset created successfully!")