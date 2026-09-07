import joblib
import re

# Load trained model
model = joblib.load("model/fake_news_model.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# Function to clean text
def clean_text(text):
    text = str(text)
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


# Take news from user
news = input("\nEnter a news article:\n")

# Clean news
cleaned_news = clean_text(news)

# Convert text into TF-IDF
news_tfidf = vectorizer.transform([cleaned_news])

# Predict
prediction = model.predict(news_tfidf)[0]

# Probability
probability = model.predict_proba(news_tfidf)[0]

# Display result
print("\n-----------------------------")
print("       PREDICTION")
print("-----------------------------")

if prediction == "FAKE":
    print("Result: ❌ FAKE NEWS")
else:
    print("Result: ✅ REAL NEWS")

print(f"Confidence: {max(probability) * 100:.2f}%")