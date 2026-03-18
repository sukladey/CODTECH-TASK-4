# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from wordcloud import WordCloud
# Download NLTK Data
nltk.download('punkt')
nltk.download('stopwords')

# Create Sample Dataset (Tweets / Reviews)
data = {
    "review": [
        "I love this product, it is amazing",
        "This is the worst service ever",
        "Very happy with the quality",
        "I hate this item",
        "Excellent product and great support",
        "Not satisfied with the purchase",
        "Absolutely fantastic experience",
        "Terrible customer service",
        "Highly recommend this product",
        "Waste of money"
    ],
    
    "sentiment": [
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative"
    ]
}

df = pd.DataFrame(data)
df.head()

# Data Preprocessing
stop_words = set(stopwords.words('english'))

def preprocess(text):
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenization
    words = word_tokenize(text)
    
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

df["clean_review"] = df["review"].apply(preprocess)

df.head()

# Convert Text into Numerical Features (TF-IDF)
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["clean_review"])

y = df["sentiment"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Machine Learning Model
model = LogisticRegression()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Sentiment Distribution Plot
sns.countplot(x="sentiment", data=df)

plt.title("Sentiment Distribution")
plt.show()

# WordCloud Visualization
text = " ".join(df["clean_review"])

wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud")
plt.show()

# Test with Custom Sentence
test_text = ["The product quality is very good"]

test_vector = vectorizer.transform(test_text)

prediction = model.predict(test_vector)

print("Sentiment:", prediction[0])
