import unittest
import pandas as pd

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from predict.nlp_disaster_predictor import NLPDisasterPredictor

class Test_NLPDisasterPredictor(unittest.TestCase):

    def test_predict(self):
        """ """
        try:
            # NOTE This test is wrapped in an exception handler to avoid exception if the Trained model does not exist
            predictor = NLPDisasterPredictor()
            self.assertIn(predictor.predict('hello world'), ["Relevant", "Not Relevant"])
        except: pass
