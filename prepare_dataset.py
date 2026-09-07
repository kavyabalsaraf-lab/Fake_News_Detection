import pandas as pd

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Add labels
fake["label"] = "FAKE"
true["label"] = "REAL"

# Combine both datasets
df = pd.concat([fake, true], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save combined dataset
df.to_csv("dataset/news.csv", index=False)

# Display information
print("Dataset created successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nFirst 5 rows:")
print(df.head())
