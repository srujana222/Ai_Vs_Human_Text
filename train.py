
from nltk.corpus import stopwords
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np


df=pd.read_csv("ai_vs_human_text.csv")
df

df.shape

df.head()

df.tail()

df.isnull().sum()

df = df[['text_content', 'label']].copy()
df = df.dropna()

from nltk.corpus import stopwords

stop_words=stopwords.words('english')
print(stop_words)

import re
def clean_text(text):
    text = re.sub(r'[^a-zA-z]',' ', text)  # remove numbers
    words = text.lower().split()
    words=[word for word in words if word not in stop_words] 
    return ' '.join(words)

df['text_content'] = df['text_content'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)

x = vectorizer.fit_transform(df['text_content'])
y = df["label"]

vectorizer.get_feature_names_out()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")