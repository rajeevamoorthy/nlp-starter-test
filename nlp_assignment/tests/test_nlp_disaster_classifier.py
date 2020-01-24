import unittest
import pandas as pd

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from train.nlp_disaster_classifier import NLPDisasterClassifier
from utils.nlp_pipeline import NLPPipeline

class Test_NLPDisasterClassifier(unittest.TestCase):

    def test_calssify(self):
        """ """
        trainer = NLPDisasterClassifier()

        df = trainer.read_data()
        # NOTE This is reusing the read_data from the production pipeline.
        # It would be better to use a test data corpus instead to check the quality of the classifier during CI

        precision, recall = trainer.train(data_corpus=df, save_model=False)

        self.assertGreater(precision, 0.75)
        self.assertGreater(recall, 0.75)
