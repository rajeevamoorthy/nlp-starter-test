
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import pandas as pd
from utils.nlp_pipeline import NLPPipeline
from config import params

class NLPDisasterPredictor:
    """ """

    def __init__(self):
        """ """
        working_dir = os.path.dirname(os.path.realpath(__file__))
        self.parent_dir = os.path.dirname(working_dir)

        self.pipeline = NLPPipeline()

        model_dict = self.pipeline.load_model(model_name=params['model_name'])
        self.classifier = model_dict.get('classifier')
        self.vectorizer = model_dict.get('vectorizer')
        self.class_labels = model_dict.get('class_labels')


    def predict(self, text):
        """ """

        sample = pd.DataFrame([[text]], columns=['text'])
        sample = self.pipeline.clean(sample, 'text')
        sample = self.pipeline.tokenize(sample, 'text')

        # print(sample)

        embedding = dict()
        embedding['test_X'] = (self.vectorizer.transform(sample['text']))
        # print(embedding)

        predicted_sentiment = self.classifier.predict(embedding['test_X']).tolist()

        # print(self.class_labels[predicted_sentiment[0]])

        return self.class_labels[predicted_sentiment[0]]


# predict = NLPDisasterPredictor()
# predict.predict('Volcano World')
