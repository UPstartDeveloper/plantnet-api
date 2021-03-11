# Serialize the Model's predictions into JSON
import json
# Prevent ImportError w/ flask
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from werkzeug.datastructures import FileStorage
# ML/Data processing
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from PIL import Image
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
        args = arg_parser.parse_args()
        image_file = args.image  # reading args from file
        image = Image.open(image_file)  # open the image
        # B: preprocess the image
        tensor_image = keras.preprocessing.image.img_to_array(image)
        resized_img = tf.image.resize(tensor_image, [256, 256])
        final_image = tf.keras.applications.inception_v3.preprocess_input(resized_img)
        # C: make a 4D tensor before we're ready to predict
        final_input = np.expand_dims(final_image, axis=0)
        # D: predict on the image data - use an outer list to make a 4D Tensor
        prediction_probabilities = model(final_input, training=False)[0]
        # E: convert the response from Tensor -> ndarray -> Python list -> JSON
        prediction_probabilities = (
            json.dumps(np.array(prediction_probabilities).tolist())
        )
        # return the classification
        return {
            "prediction": prediction_probabilities
        }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
