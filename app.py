# =========================================================
#           FAKE NEWS DETECTION SYSTEM
#        Using Machine Learning and NLP
# =========================================================

# =========================================================
# Import Libraries
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================================================
# STEP 1 : DATA COLLECTION
# =========================================================

# Load Fake and Real News Dataset

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# =========================================================
# STEP 2 : DATA PREPROCESSING
# =========================================================

# Assign Labels
# Fake News = 0
# Real News = 1

fake["label"] = 0
true["label"] = 1

# Combine Dataset
data = pd.concat([fake, true])

# Shuffle Dataset
data = data.sample(frac=1)

# =========================================================
# STEP 3 : TEXT CLEANING (NLP)
# =========================================================

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'\W', ' ', text)

    return text

# Apply Cleaning
data["text"] = data["text"].apply(clean_text)

# =========================================================
# STEP 4 : FEATURE EXTRACTION
# =========================================================

# Features
x = data["text"]

# Labels
y = data["label"]

# Convert text into numerical values using TF-IDF

vectorizer = TfidfVectorizer(stop_words='english')

x = vectorizer.fit_transform(x)

# =========================================================
# STEP 5 : TRAIN TEST SPLIT
# =========================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# STEP 6 : MODEL TRAINING
# =========================================================

# Logistic Regression Model

model = LogisticRegression(max_iter=1000)

# Train Model
model.fit(x_train, y_train)

# =========================================================
# STEP 7 : PREDICTION
# =========================================================

pred = model.predict(x_test)

# =========================================================
# STEP 8 : MODEL EVALUATION
# =========================================================

# Accuracy

accuracy = accuracy_score(y_test, pred)

print("\n===================================")
print(" FAKE NEWS DETECTION SYSTEM ")
print("===================================")

print("\nAccuracy :", round(accuracy * 100, 2), "%")

print("\n0 = Fake News")
print("1 = Real News")

# =========================================================
# GRAPH 1 : FAKE VS REAL NEWS
# =========================================================

fake_count = len(data[data["label"] == 0])

real_count = len(data[data["label"] == 1])

plt.bar(
    ["Fake News", "Real News"],
    [fake_count, real_count],
    color=["red", "green"]
)

plt.title("Fake vs Real News")

plt.xlabel("News Type")

plt.ylabel("Number of News")

plt.show()

# =========================================================
# GRAPH 2 : MODEL ACCURACY
# =========================================================

accuracy_value = accuracy * 100

error_value = 100 - accuracy_value

plt.pie(
    [accuracy_value, error_value],
    labels=["Accuracy", "Error"],
    autopct='%1.1f%%'
)

plt.title("Model Accuracy")

plt.show()

# =========================================================
# GRAPH 3 : PREDICTION RESULTS GRAPH
# =========================================================

cm = confusion_matrix(y_test, pred)

correct_fake = cm[0][0]
wrong_fake = cm[0][1]

wrong_real = cm[1][0]
correct_real = cm[1][1]

labels = ["Fake News", "Real News"]

correct = [correct_fake, correct_real]

wrong = [wrong_fake, wrong_real]

plt.figure(figsize=(6,5))

plt.bar(
    labels,
    correct,
    label="Correct Prediction"
)

plt.bar(
    labels,
    wrong,
    bottom=correct,
    label="Wrong Prediction"
)

plt.title("Prediction Results")

plt.ylabel("Number of News")

plt.legend()

plt.show()

# Axis Labels
plt.xticks([0,1], ["Fake", "Real"])
plt.yticks([0,1], ["Fake", "Real"])

# Numbers inside boxes
for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha='center',
            va='center',
            color='black',
            fontsize=14
        )

plt.colorbar()

plt.show()

# =========================================================
# STEP 9 : USER INPUT PREDICTION
# =========================================================

news = input("\nEnter News Text : ")

# Clean Input
news = clean_text(news)

# Convert Text to TF-IDF Vector
news_vector = vectorizer.transform([news])

# Predict News
output = model.predict(news_vector)

# =========================================================
# FINAL RESULT
# =========================================================

if output[0] == 0:

    print("\nPrediction : Fake News")

else:

    print("\nPrediction : Real News")

# =========================================================
# END OF PROJECT
# =========================================================