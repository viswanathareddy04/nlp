# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 16:22:45 2021

@author: injav
"""

import  nltk.tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

file_contents = open("english-content.txt", 'r').read()
sentences = nltk.sent_tokenize(file_contents)
lemma = WordNetLemmatizer()

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemma.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = " ".join(words)