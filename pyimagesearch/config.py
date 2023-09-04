# import the necessary packages
import os

PRUNING = True
VIDEORESULT = False
DATASETXML = "dataset_full"
BASE_PATH = 'dataset'

# hyperparameter training
INIT_LR = 1e-4
NUM_EPOCHS = 5 # 5 50
BATCH_SIZE = 16 # 256

# define the number10of max proposals used when running selective
# search for (1) gathering training data and (2) performing inference
MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

# define the maximum number of positive and negative images to be
# generated from each image
MAX_POSITIVE = 30
MAX_NEGATIVE = 10

# initialize the input dimensions to the network
INPUT_DIMS = (224, 224)

# define the path to the output model and label binarizer
MODEL_PATH = "roadCrackDetector.h5"
# MODEL_PATH = "model_tflite.tflite"
ENCODER_PATH = "label_encoder.pickle"

# define the minimum probability required for a positive prediction
# (used to filter out false-positive predictions)
MIN_PROBA = 0.75
