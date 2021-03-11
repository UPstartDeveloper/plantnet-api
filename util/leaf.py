# Processing Input and Output Data
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras


# define the labels for the leave health
LEAF_LABELS = np.array([
    # entries MUST be formatted as "<species>_<condition>"
    'Strawberry_healthy',
    'Grape_Black rot',
    'Potato_Early blight',
    'Blueberry_healthy',
    'Corn_healthy',
    'Tomato_Target Spot',
    'Peach_healthy',
    'Potato_Late blight',
    'Tomato_Late blight',
    'Tomato_mosaic virus',
    'Bell pepper_healthy',
    'Orange_Haunglongbing (aka Citrus greening)',
    'Tomato_Leaf Mold',
    'Grape_Leaf blight (aka Isariopsis leaf spot)',
    'Cherry_Powdery mildew',
    'Apple_rust',
    'Tomato_Bacterial spot',
    'Grape_healthy',
    'Tomato_Early blight',
    'Corn_Common rust',
    'Grape_Black Measles',
    'Raspberry_healthy',
    'Tomato_healthy',
    'Cherry_healthy',
    'Tomato_Yellow Leaf Curl Virus',
    'Apple_scab',
    'Corn_Northern Leaf Blight',
    'Tomato_Two-spotted spider mite',
    'Peach_Bacterial spot',
    'Bell pepper_Bacterial spot',
    'Tomato_Septoria leaf spot',
    'Squash_Powdery mildew',
    'Corn_Gray leaf spot',
    'Apple_Black rot',
    'Apple_healthy',
    'Strawberry_Leaf scorch',
    'Potato_healthy',
    'Soybean_healthy'
])


def get_image(arg_parser):
    '''Returns a Pillow Image given the uploaded leaf image.'''
    args = arg_parser.parse_args()
    image_file = args.image  # reading args from file
    return Image.open(image_file)  # open the image


def preprocess_image(image):
    """Converts a PIL.Image into a Tensor of the 
    right dimensions for Inception V3.
    """
    tensor_image = keras.preprocessing.image.img_to_array(image)
    resized_img = tf.image.resize(tensor_image, [256, 256])
    return tf.keras.applications.inception_v3.preprocess_input(resized_img)


def predict_leaf_health(model, image):
    """Returns the most likely class for the image 
    according to the output of the model.

    Parameters:
    model(keras.Model): the CNN making the prediction
    image (tf.Tensor): a 3D tensor representing a color image of the leaf

    Source: the "New Plant Diseases Dataset": https://tinyurl.com/dzav422a

    Returns: dict: the label and the models confidence associated thereof it
                   are included as fields
    """
    # make a 4D tensor before we're ready to predict
    final_input = np.expand_dims(image, axis=0)
    # predict on the image data - pull out the probabilities for each class
    prediction_probabilities = model(final_input, training=False)[0]
    # get the prediction label
    index_highest_proba = np.argmax(prediction_probabilities)
    label = str(LEAF_LABELS[index_highest_proba])
    # get the prediction probability
    confidence = float(prediction_probabilities[index_highest_proba])
    # return the output as a JSON string
    output = {
        "label": label, 
        "confidence": confidence
    }
    return output
