# Make a flask API for our DL Model
import werkzeug
# This next lime Prevents ImportErrors - flask-restplus currently gets broken by 
# Werkzeug 1.0.0. Alternatively, you could also downgrade werkzeug 
# (the package Python uses to interface with an HTTP web server)
# Note: line 6 must happen BEFORE you import Flask to be able to work!
werkzeug.cached_property = werkzeug.utils.cached_property

import tensorflow as tf
import tensorflow.keras as keras
from keras.preprocessing.image import img_to_array
from flask_restplus import Api, Resource, fields
from flask import Flask, request, jsonify
import numpy as np
from werkzeug.datastructures import FileStorage
from PIL import Image


application = app = Flask(__name__)
api = Api(app, version='1.0', title='MNIST Classification', description='CNN for Mnist')
ns = api.namespace('Make_School', description='Methods')

# use Flask-RESTPlus argparser to help make predictions on the input requests 
arg_parser = api.parser()
arg_parser.add_argument('image_location', required=True)

# Model reconstruction from Protocol Buffer format - do this only once!
model = tf.keras.load_model('plantnet')

@ns.route('/prediction')
class CNNPrediction(Resource):
    """Uploads your data to the CNN"""
    @api.doc(parser=arg_parser, description='Tell me where your image is in S3')
    def post(self):
        pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
