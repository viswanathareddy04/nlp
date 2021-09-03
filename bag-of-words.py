# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 21:01:59 2021

@author: injav
"""

import nltk.tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer

file_contents = open('english-content.txt').read()
ps = PorterStemmer()
wnl = WordNetLemmatizer()
sentences = nltk.sent_tokenize(file_contents)
stem_corpus = []
lemm_corpus = []
for i in range(len(sentences)):
    words =  re.sub('[^a-zA-Z]',' ',sentences[i])
    words = words.lower()
    words = words.split()
    stem_words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    lemm_words = [wnl.lemmatize(word) for word in words if word not in stopwords.words('english')]
    lemm_words = " ".join(lemm_words)
    stem_words = " ".join(stem_words)
    stem_corpus.append(stem_words)
    lemm_corpus.append(lemm_words)


countVectorizer = CountVectorizer(max_features=1500)
stemVectorX = countVectorizer.fit_transform(stem_corpus).toarray()
lemmVectorY = countVectorizer.fit_transform(lemm_corpus).toarray()
