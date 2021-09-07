# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 16:03:25 2021

@author: injav
"""

import nltk.tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

file_contents = open("data/english-content.txt").read()

sentences = nltk.sent_tokenize(file_contents)
stemmer =  PorterStemmer()
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)   
    
