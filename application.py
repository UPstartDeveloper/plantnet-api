# Prevent ImportError w/ flask
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from werkzeug.datastructures import FileStorage
# ML/Data processing
import tensorflow.keras as keras
import numpy as np
# RESTful API packages
from flask_restplus import Api, Resource
from flask import Flask
# Utility Functions
from util import leaf

application = app = Flask(__name__)
api = Api(app, version="1.0", title="PlantNet API", 
        description="Identifying Leaf Disease via Deep Learning")
ns = api.namespace(
    "Diagnosis", 
    description="Represents the health states of a leaf understood by the AI."
)

# use Flask-RESTPlus argparser to help make predictions on the input requests
arg_parser = api.parser()
arg_parser.add_argument('image', location='files',
                           type=FileStorage, required=True)

# Model reconstruction - using retrained weights from Inception V3
with open("./plantnet/inceptionModelArchitecture.json") as f:
    model = keras.models.model_from_json(f.read())
    model.load_weights("./plantnet/inception_model_weights.h5")


@ns.route("/prediction")
class CNNPrediction(Resource):
    """Takes in the image, to pass to the CNN"""
    @api.doc(parser=arg_parser, 
             description="Let the AI predict if a leaf is healthy.")
    def post(self):
        # A: get the image
        image = leaf.get_image(arg_parser)
        # B: preprocess the image
        final_image = leaf.preprocess_image(image)
        # C: make the prediction
        prediction = leaf.predict_leaf_health(model, final_image)
        # return the classification
        return prediction


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
