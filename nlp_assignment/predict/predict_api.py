import falcon
from falcon import uri

from nlp_assignment.predict.nlp_disaster_predictor import NLPDisasterPredictor

class PredictApi:
    """
    """

    def on_get(self, req, resp):
        """
        REST endpoint exposing the predict API.
        """
        predict = NLPDisasterPredictor()

        qs = uri.parse_query_string(req.query_string)
        text = qs.get('text')

        resonse_body = {
            'text': text,
            'prediction': predict.predict(text)
        }

        resp.media = resonse_body
        resp.status = falcon.HTTP_200
