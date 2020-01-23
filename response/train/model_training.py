# from tensorflow import keras
# import gensim
import nltk
# import sklearn
# import numpy as np
# import matplotlib

import re, os, pickle
# import codecs
# import itertools
import sqlite3
import pandas as pd

from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score


class ModelTraining:
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
        self.working_dir = os.path.dirname(os.path.realpath(__file__))

    def read_data(self):
        '''
        Input: None
        Output: dataframe
        '''
        parent_dir = os.path.dirname(self.working_dir)

        conn = sqlite3.connect(parent_dir + "/db.sqlite")
        df = pd.read_sql_query("select * from data;", conn)

        conn.close()

        return df

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

    def get_embedding(self, X_train, X_test, y_train, y_test):
        '''
        Input: dataframe
        Output: dataframe

        A simple TFDIF model is used for feature extraction
        - More sophistication can be implemented here to improve the model
        '''
        vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w+')
        embedding = dict()
        embedding["train"] = (vectorizer.fit_transform(X_train), y_train)
        embedding["test"]  = (vectorizer.transform(X_test), y_test)

        return embedding

    def get_metrics(self, y_test, y_predicted):
        ''' '''
        precision = precision_score(y_test, y_predicted, pos_label=None, average='weighted')
        recall = recall_score(y_test, y_predicted, pos_label=None, average='weighted')

        print("precision = %.3f, recall = %.3f" % (precision, recall))
        return precision, recall # Use this in test

    def save_model(self, classifier):
        ''' '''
        pickled_classifier = pickle.dumps(classifier)

        with open(self.working_dir + "/classifier.pickle", "wb") as trained_model:
            trained_model.write(pickled_classifier)

        return None

    def train(self):
        ''' Training pipeline
        - read data
        - clean
        - tokenize
        - split inputs into test/train data sets
        - Feature extraction
        - choose a classifier and fit()
        - Serialize model and save to disk
        '''
        survey = self.read_data()
        survey = self.clean(survey, "text")
        survey = self.tokenize(survey, "text")

        # Split input data into test
        X_train, X_test, y_train, y_test = train_test_split(survey["text"], survey["class_label"], test_size=0.2, random_state=40)

        embedding = self.get_embedding(X_train, X_test, y_train, y_test)

        classifier = MultinomialNB()
        classifier.fit(*embedding["train"])

        y_predict = classifier.predict(embedding["test"][0])
        precision, recall = self.get_metrics(embedding["test"][1], y_predict)

        if precision >= 0.75 and recall >= 0.75:
            self.save_model(classifier)
        else:
            raise Exception('Model is not acceptable')


# train = ModelTraining()
# train.train()

