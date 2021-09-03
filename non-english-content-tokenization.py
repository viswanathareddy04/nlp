# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 07:27:48 2021

@author: injav
"""
import nltk.data
tokenize_spanish = nltk.data.load('tokenizers/punkt/spanish.pickle')
file_contents = open(r'spanish-content.txt',  encoding="utf8").read()
print(tokenize_spanish.tokenize(file_contents)[0])
