
import re, os, pickle
from datetime import datetime

import sqlite3
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score

from nlp_pipeline import NLPPipeline

class NLPDisasterClassifier:
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

    def read_data(self):
        '''
        Input: None
        Output: dataframe
        '''

        conn = sqlite3.connect(self.parent_dir + "/db.sqlite")
        df = pd.read_sql_query("select * from data;", conn)

        conn.close()

        return df

    def get_metrics(self, y_expected, y_predicted):
        '''
        Input: Expected and predicted values
        Output: Precision and recall
        - To implement more stringent validation checks.
        '''
        precision = precision_score(y_expected, y_predicted, pos_label=None, average='weighted')
        recall = recall_score(y_expected, y_predicted, pos_label=None, average='weighted')

        print(f"precision = {precision:.3f}, recall = {recall:.3f}")
        return precision, recall # Use this in test

    def get_embedding(self, X_train, X_test, y_train, y_test):
        '''
        Input: dataframe
        Output: dataframe

        A simple TFDIF model is used for feature extraction
        - More sophistication can be implemented here to improve the model
        '''

        return embedding

    def save_model(self, vectorizer, classifier):
        '''
        Input: Trained classifier model
        Output: None
        - Backup the previous model (if it exists)
        - Save current model as a pickled
        '''
        model_pickle = self.parent_dir + "/nlp_disaster_model.pickle"

        # try:
        #     modifiedTime = os.path.getmtime(model_pickle)
        #     timestamp = datetime.fromtimestamp(modifiedTime).strftime("%Y%m%d%H%M%S")

        #     os.rename(model_pickle, model_pickle + "_" + timestamp)
        # except Exception as e: print(e)

        with open(model_pickle, "wb") as out_file:
            out_file.write(pickle.dumps({
                "classifier": classifier,
                "vectorizer": vectorizer,
                "class_labels": {
                    1: "Relevant",
                    0: "Not Relevant",
                }
            }))

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
        pipeline = NLPPipeline()

        survey = self.read_data()
        survey = pipeline.clean(survey, "text")
        survey = pipeline.tokenize(survey, "text")
        vectorizer = pipeline.get_vectorizer()

        # Split input data into test
        X_train, X_test, y_train, y_test = train_test_split(survey["text"], survey["class_label"], test_size=0.2, random_state=40)

        # print(X_test)
        embedding = dict()
        embedding["train"] = (vectorizer.fit_transform(X_train), y_train)
        embedding["test"]  = (vectorizer.transform(X_test), y_test)

        classifier = MultinomialNB()
        classifier.fit(*embedding["train"])

        y_predict = classifier.predict(embedding["test"][0])

        # print(embedding["test"][0])
        # print(y_predict.tolist())

        precision, recall = self.get_metrics(embedding["test"][1], y_predict)

        if precision >= 0.75 and recall >= 0.75:
            self.save_model(vectorizer, classifier)
        else:
            raise Exception(f'Model is not acceptable: precision({precision}), recall({recall})')


train = NLPDisasterClassifier()
train.train()

