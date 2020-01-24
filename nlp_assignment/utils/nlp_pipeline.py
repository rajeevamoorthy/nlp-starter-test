
import os, pickle, re

from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


class NLPPipeline:
    """
    Each step in the model generation pipeline is implemented as a method in this class to allow for extendability.
    - read data (multiple sources)
    - clean ( any number of )
    - tokenize input strings into component pieces (words)
    - Feature extraction
    - choose classifier and fit model
    - pickle and save model to disk
    """

    def __init__(self):
        """ """
        working_dir = os.path.dirname(os.path.realpath(__file__))
        self.parent_dir = os.path.dirname(working_dir)

    def clean(self, df, field):
        """
        Input: dataframe
        Output: dataframe
        """
        df[field] = df[field].str.lower()
        df[field] = df[field].apply(lambda elem: re.sub(r'http\S+', '', elem))  # get rid of URLs
        return df

    def tokenize(self, df, field):
        """
        Input: dataframe
        Output: dataframe
        """
        tokenizer = RegexpTokenizer(r'\w+')

        df['tokens'] = df[field].apply(tokenizer.tokenize)
        return df

    def get_vectorizer(self):
        """ """
        return TfidfVectorizer(analyzer='word', token_pattern=r'\w+')

    def save_model(self, vectorizer, classifier, model_name):
        """
        Input: Trained classifier model
        Output: None
        - Backup the previous model (if it exists)
        - Save current model as a pickled
        """
        model_pickle = self.parent_dir + '/' + model_name

        with open(model_pickle, 'wb') as out_file:
            out_file.write(pickle.dumps({
                'classifier': classifier,
                'vectorizer': vectorizer,
                'class_labels': {
                    1: 'Relevant',
                    0: 'Not Relevant',
                }
            }))

        return None

    def load_model(self, model_name):
        """ """
        model_pickle = self.parent_dir + '/' + model_name

        with open(model_pickle, 'rb') as in_file:
            model_dict = pickle.loads(in_file.read())

        return model_dict
