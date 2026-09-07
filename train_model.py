import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load processed dataset
df = pd.read_csv("dataset/processed_news.csv")

# Handle missing values
df["clean_text"] = df["clean_text"].fillna("")

# Remove empty text
df = df[df["clean_text"].str.strip() != ""]

# Input and output
X = df["clean_text"]
y = df["label"]

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_tfidf = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dataset split completed!")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
print("\nTraining model...")
model.fit(X_train, y_train)

print("Model training completed!")

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy * 100:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model/fake_news_model.pkl")

# Save TF-IDF vectorizer
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nModel saved successfully!")
print("model/fake_news_model.pkl")
print("model/tfidf_vectorizer.pkl")