# Import necessary libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

# Download NLTK data
nltk.download("vader_lexicon")

df_users = pd.read_csv("users.csv")
df_fusers = pd.read_csv("fusers.csv")

# Add fake account
isNotFake = np.zeros(3474)
isFake = np.ones(3351)

# Adding is fake or not column to make predictions for it
df_fusers["isFake"] = isFake
df_users["isFake"] = isNotFake

df_allUsers = pd.concat([df_fusers, df_users], ignore_index=True)
df_allUsers.columns = df_allUsers.columns.str.strip()

# Shuffle the whole data
df_allUsers = df_allUsers.sample(frac=1).reset_index(drop=True)

Y = df_allUsers.isFake
df_allUsers.drop(["isFake"], axis=1, inplace=True)

# Select only numeric columns for the profile report
X = df_allUsers.select_dtypes(include=np.number)

Y.reset_index(drop=True, inplace=True)


def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)["compound"]

    if sentiment_score >= 0.05:
        return "Positive"
    elif sentiment_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def detect_threats(text):
    cybersecurity_keywords = [
        "phishing",
        "malware",
        "data breach",
        "vulnerability",
        "cyber attack",
    ]

    for keyword in cybersecurity_keywords:
        if keyword in text.lower():
            return True

    return False


# Function to train a basic machine learning model for threat detection
def train_threat_model(data, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )

    threat_model = MultinomialNB()
    threat_model.fit(X_train, y_train)

    predictions = threat_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return threat_model, accuracy


# Example usage
sample_text = "The new cybersecurity measures are impressive!"
sentiment_result = analyze_sentiment(sample_text)
threat_result = detect_threats(sample_text)

print(f"Sentiment Result: {sentiment_result}")
print(f"Threat Result: {threat_result}")




# Generate the profile report
profile = ProfileReport(X, title="Pandas Profiling Report", config_file="config.yaml")
profile.to_file(output_file="output.html")
