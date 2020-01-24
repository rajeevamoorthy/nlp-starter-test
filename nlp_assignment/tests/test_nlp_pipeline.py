import unittest
import pandas as pd

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.nlp_pipeline import NLPPipeline

class Test_NLPPipeline(unittest.TestCase):

    def test_clean(self):
        """ """
        pipeline = NLPPipeline()

        df = pd.DataFrame([['Hello World']], columns=['text'])
        clean_df = pipeline.clean(df, 'text')

        self.assertEqual(clean_df['text'][0], 'hello world')

    def test_tokenize(self):
        """ """
        pipeline = NLPPipeline()

        df = pd.DataFrame([['Hello World']], columns=['text'])
        tokenized_df = pipeline.tokenize(df, 'text')

        self.assertEqual(tokenized_df['tokens'][0], ['Hello', 'World'])

    def test_get_vectorizer(self):
        """
        """
        pipeline = NLPPipeline()

        vectorizer = pipeline.get_vectorizer()

        self.assertIsNotNone(vectorizer)
