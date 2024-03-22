#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf

# Define the feature extractor using MobileNetV2
def feature_extractor(inputs):
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    features = base_model(inputs)
    return features

# Define the dense layers
def dense_layers(x):
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    flatten_layer = tf.keras.layers.Flatten()
    dense_1 = tf.keras.layers.Dense(1024, activation='relu')
    dense_2 = tf.keras.layers.Dense(512, activation='relu')

    x = global_average_layer(x)
    x = flatten_layer(x)
    x = dense_1(x)
    x = dense_2(x)
    return x

# Define the bounding box regression layer
def bounding_box_regression(x):
    bounding_box_regression_output = tf.keras.layers.Dense(4, name='bounding_box')(x)
    return bounding_box_regression_output

# Define the final model
def final_model(inputs):
    features = feature_extractor(inputs)
    dense_output = dense_layers(features)
    bounding_box_output = bounding_box_regression(dense_output)
    model = tf.keras.Model(inputs=inputs, outputs=bounding_box_output)
    return model

# Define and compile the model
def define_and_compile_model():
    # Define the input layer
    inputs = tf.keras.Input(shape=(224, 224, 3))

    # Create the model
    model = final_model(inputs)

    # Compile the model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model

# Define your model
model = define_and_compile_model()

# Print model summary
model.summary()


# In[ ]:




