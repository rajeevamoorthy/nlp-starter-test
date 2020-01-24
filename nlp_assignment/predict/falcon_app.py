import falcon
from falcon_swagger_ui import register_swaggerui_app
import pathlib

from nlp_assignment.predict.predict_api import PredictApi

app = falcon.API()
app.add_route('/predict', PredictApi())

SWAGGERUI_URL = '/swagger'  # without trailing slash
SCHEMA_URL = '/static/v1/swagger.json'
STATIC_PATH = pathlib.Path(__file__).parent / 'static'
#
# @see: http://falcon.readthedocs.io/en/stable/api/api.html#falcon.API.add_static_route
app.add_static_route('/static', str(STATIC_PATH))

page_title = 'Falcon Swagger Doc'

register_swaggerui_app(
    app,
    SWAGGERUI_URL,
    SCHEMA_URL,
    page_title=page_title,
    config={'supportedSubmitMethods': ['get'], }
)
