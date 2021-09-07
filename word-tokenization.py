# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 07:24:09 2021

@author: injav
"""

import nltk.tokenize

file_contents = open(r'data/english-content.txt', 'r').read()
word_tokens = nltk.word_tokenize(file_contents)
print(word_tokens)