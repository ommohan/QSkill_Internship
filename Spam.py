
##To Build a classifier that distinguishes between spam and non-spam (ham) emails

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("mail_data.csv")


nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
df['Message'] = df['Message'].str.lower()


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Message']).toarray()


y = df['Category']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = MultinomialNB()
model.fit(X_train,y_train)

log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score,f1_score,precision_score


print("Accuracy:", accuracy_score(y_test,y_pred))
print("Naive Bayes Result:")
#print("Model score:",model.score(X_test,y_test))
print("Precision score:",precision_score(y_test,y_pred,average='weighted'))
print("F1 Score:",f1_score(y_test,y_pred,average='weighted'))
print("logistic regression score:",log_reg.score(X_test,y_test))
