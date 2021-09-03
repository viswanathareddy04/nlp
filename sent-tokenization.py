# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 06:11:40 2021

@author: injav
"""


import nltk.tokenize

file_contents = open(r'english-content.txt', 'r').read()
sent_tokens = nltk.sent_tokenize(file_contents)
print(sent_tokens)