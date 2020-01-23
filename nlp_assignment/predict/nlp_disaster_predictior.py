
import re, os, pickle, sqlite3
import pandas as pd
from nlp_pipeline import NLPPipeline

class NLPDisasterPredictor:
    ''' '''

    def __init__(self):
        ''' '''
        working_dir = os.path.dirname(os.path.realpath(__file__))
        self.parent_dir = os.path.dirname(working_dir)

        model_pickle = self.parent_dir + "/nlp_disaster_model.pickle"

        with open(model_pickle, "rb") as in_file:
            model_dict = pickle.loads(in_file.read())

        self.classifier = model_dict.get('classifier')
        self.vectorizer = model_dict.get('vectorizer')
        self.class_labels = model_dict.get('class_labels')

    def predict(self, text):
        ''' '''
        pipeline = NLPPipeline()

        conn = sqlite3.connect(self.parent_dir + "/db.sqlite")
        sample = pd.read_sql_query("select * from data;", conn)
        conn.close()

        sample = pd.DataFrame([[text]], columns=['text'])
        sample = pipeline.clean(sample, "text")
        sample = pipeline.tokenize(sample, "text")

        # print(sample)

        embedding = dict()
        embedding["test_X"] = (self.vectorizer.transform(sample['text']))
        # print(embedding)

        predicted_sentiment = self.classifier.predict(embedding['test_X']).tolist()

        # print(self.class_labels[predicted_sentiment[0]])

        return self.class_labels[predicted_sentiment[0]]


predict = NLPDisasterPredictor()
predict.predict('Volcano World')
