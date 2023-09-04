import tensorflow as tf
from pyimagesearch import config

# Define the paths to the input and output files
h5_model_path = config.MODEL_PATH
tflite_model_path = './model_tflite.tflite'

# Load the h5 model
h5_model = tf.keras.models.load_model(h5_model_path)

# Convert the h5 model to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(h5_model)
tflite_model = converter.convert()

# Save the tflite model to a file
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print("Model converted successfully!")