import pandas as pd
import re

# Load combined dataset
df = pd.read_csv("dataset/news.csv")

# Remove missing values
df = df.dropna(subset=["text"])

# Function to clean text
def clean_text(text):
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply preprocessing
df["clean_text"] = df["text"].apply(clean_text)

# Save processed dataset
df.to_csv("dataset/processed_news.csv", index=False)

print("Preprocessing completed!")

print("\nOriginal text:")
print(df["text"].iloc[0][:300])

print("\nCleaned text:")
print(df["clean_text"].iloc[0][:300])

print("\nDataset shape:")
print(df.shape)