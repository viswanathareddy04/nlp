# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 05:40:48 2021

@author: injav
"""

import nltk.tokenize
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
import re

file_contents = open(r'data\english-content.txt').read()


text = re.sub(r'\[[0-9]*\]',' ',file_contents)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
     sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
     
model = Word2Vec(sentences, min_count=1)
words = model.wv

# Finding Word Vectors
vector = model.wv['rate']

# Most similar words
similar = model.wv.most_similar('banking')