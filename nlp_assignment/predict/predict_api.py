import falcon, json
from nlp_disaster_predictior import NLPDisasterPredictor

class PredictApi:

    def on_post(self, req, resp):
        """Handles POST requests"""
        predict = NLPDisasterPredictor()

        data = req.media

        resonse_body = {
            'input text': data["input"],
            'prediction': predict.predict(data["input"])
        }

        resp.media = resonse_body
        resp.status = falcon.HTTP_200

api = falcon.API()
api.add_route('/predict', PredictApi())
