
import re, os, pickle
from datetime import datetime

import sqlite3
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score


class NLPPipeline:
    '''
    Each step in the model generation pipeline is implemented as a method in this class to allow for extendability.
    - read data (multiple sources)
    - clean ( any number of )
    - tokenize input strings into component pieces (words)
    - Feature extraction
    - choose classifier and fit model
    - pickle and save model to disk
    '''

    def __init__(self):
        ''' '''
        working_dir = os.path.dirname(os.path.realpath(__file__))
        self.parent_dir = os.path.dirname(working_dir)


    def clean(self, df, field):
        '''
        Input: dataframe
        Output: dataframe
        '''
        df[field] = df[field].str.lower()
        df[field] = df[field].apply(lambda elem: re.sub(r"http\S+", "", elem))  # get rid of URLs
        return df


    def tokenize(self, df, field):
        '''
        Input: dataframe
        Output: dataframe
        - This is probably generic for NLP pipelines.
        '''
        tokenizer = RegexpTokenizer(r'\w+')

        df["tokens"] = df[field].apply(tokenizer.tokenize)
        return df

    def get_vectorizer(self):
        ''' '''
        return TfidfVectorizer(analyzer='word', token_pattern=r'\w+')

