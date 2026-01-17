import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load dataset
df = pd.read_csv("data/train.csv")

# 2. Create binary target
df["is_toxic"] = (
    df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]
    .sum(axis=1) > 0
).astype(int)

X = df["comment_text"]
y = df["is_toxic"]

# 3. Train-test split (IMPORTANT: stratify!)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 6. Predict on test set
y_pred = model.predict(X_test_tfidf)

# 7. Evaluation (THIS IS THE MOST IMPORTANT PART)
print(classification_report(y_test, y_pred))
