# Serialize the Model's predictions into JSON
import json
# Prevent ImportError w/ flask
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from werkzeug.datastructures import FileStorage
# ML/Data processing
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as keras
import numpy as np
from PIL import Image
# RESTful API packages
from flask_restplus import Api, Resource, fields
from flask import Flask


application = app = Flask(__name__)
api = Api(app, version="1.0", title="PlantNet API", 
        description="Inference on Leaf Images")
ns = api.namespace("Diagnose Leaf Condition", 
                    description="Works best on images of 1 leaf.")

# use Flask-RESTPlus argparser to help make predictions on the input requests
arg_parser = api.parser()
arg_parser.add_argument('image', location='files',
                           type=FileStorage, required=True)

# Model reconstruction from Protocol Buffer format - do this only once!
model = tf.keras.models.load_model("plantnet")


@ns.route("/prediction")
class CNNPrediction(Resource):
    """Takes in the image, to pass to the CNN"""
    @api.doc(parser=arg_parser, description="Tell me where your image is in S3")
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
