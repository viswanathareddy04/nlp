# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 06:46:58 2021

@author: injav
"""
import re
import pandas as pd
import nltk.tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

sms_messages = pd.read_csv('data/sms_spam_list.csv', sep=',')
df = pd.DataFrame(sms_messages)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

porter_stemmer = PorterStemmer()
corpus = []

for i in range(len(df)):
    words = re.sub('[^a-zA-Z]', ' ', df['v2'][i])
    words = words.lower()
    words = words.split()
    words = [porter_stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    words = ' '.join(words)
    corpus.append(words)
    

cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(df['v1'])
y = y.iloc[:,1].values

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# Training Model 
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred = spam_detect_model.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
    


