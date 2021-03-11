import numpy as np


def convert_probabilities(prediction_probabilities):
    """Returns the most likely class for the image 
    according to the output of the model

    Source: the "New Plant Diseases Dataset": https://tinyurl.com/dzav422a

    Parameters: array-like, contains floating point numbers betwee 0-1

    Returns: List[str, float] - the label and its probability
    """
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
    # get the prediction label
    index_highest_proba = np.argmax(prediction_probabilities)
    label = str(LEAF_LABELS[index_highest_proba])
    # B: get the prediction probability
    confidence = float(prediction_probabilities[index_highest_proba])
    return list([label, confidence])
