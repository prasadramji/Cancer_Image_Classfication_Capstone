import joblib
import os
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input


def create_model():
    """
    creates the best model ie the attention model
    and returns the model after loading the weights
    """
    Image_Size=[224,224]

    resnet=Sequential()

    resnet.add(ResNet101(input_shape=Image_Size+[3],weights='imagenet',include_top=False,pooling='average'))

    # This is to ensure the base won't be trained again
    for layer in resnet.layers:
        layer.trainable=False
        
    flatten=Flatten()(resnet.output)
    dense=Dense(256,activation='relu')(flatten)
    dense=Dense(128,activation='relu')(dense)
    dense=Dropout(0.2)(dense)   # We add a dropout here to prevent overfitting
    prediction=Dense(5,activation='softmax')(dense)

    resmodel=Model(inputs=resnet.input,outputs=prediction)
    resmodel.load_weights('./models/resnet101_weights.h5')
    return resmodel

def predict_Category(image1, model):
    """given image1 filepaths returns the predicted caption
    """

    # img = image.load_img(img_path, target_size=(224,224))
    # x = image.img_to_array(image1)
    x = np.expand_dims(image1, axis=0)
    x = preprocess_input(x)

    return model.predict(x)
