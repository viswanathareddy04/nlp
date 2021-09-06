# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 08:23:37 2021

@author: injav
"""

import nltk.tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer

file_contents = open("english-content.txt", 'r').read()
lemma = WordNetLemmatizer()
sentences = nltk.sent_tokenize(file_contents)
corpus =[]
for i in range(len(sentences)):
    words = re.sub("[^a-zA-Z]", ' ', sentences[i])
    words = words.lower()
    words = words.split()
    words = [lemma.lemmatize(word) for word in words if word not in set(stopwords.words("english"))]
    words = " ".join(words)
    corpus.append(words)
    
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
