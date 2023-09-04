# USAGE
# python fine_tune_rcnn.py

# import the necessary packages
import tensorflow as tf
import tempfile
from pyimagesearch import config
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.models import clone_model
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.utils import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow_model_optimization as tfmot
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
import pandas as pd
import tensorflow as tf
from keras.utils.vis_utils import plot_model

# =================
# PENS 2023
# =================

if not config.PRUNING:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu,True)
if config.PRUNING:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# GPU setting
# When pruning use it to disable GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#        tf.config.experimental.set_memory_growth(gpu,True)

# os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=\'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\''

# When without pruning use it to enable GPU and limit GPU
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.set_logical_device_configuration(
#         gpus[0],
#         [tf.config.LogicalDeviceConfiguration(memory_limit=3072)])
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)
# tf.debugging.set_log_device_placement(True)

# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# construct the argument parser and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = config.INIT_LR
# print(0.0001 * tf.math.exp(-0.1))
# import time
# time.sleep(100)
def scheduler(epoch, INIT_LR):
  if epoch < 20 :
    return INIT_LR
  else:
    return INIT_LR * tf.math.exp(-0.1)

EPOCHS = config.NUM_EPOCHS   # 5
BS = config.BATCH_SIZE

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class labels
print("[INFO] loading images...")
imagePaths = list(paths.list_images(config.BASE_PATH))
data = []
labels = []

# loop over the image paths
for idx, imagePath in enumerate(imagePaths):
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the input image (224x224) and preprocess it
    image = load_img(imagePath, target_size=config.INPUT_DIMS)
    image = img_to_array(image)
    image = preprocess_input(image)

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

print("After Load")
# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)
number_label = len(set(labels)) #set biar unique value yang diambil
# one hot encoding multiclass
lb = LabelEncoder() #ubah ke numerik
labels = lb.fit_transform(labels)
labels = to_categorical(labels) #[0 0 1 0 0] [1 0 0 0 0]

print(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.3, stratify=labels, random_state=42) #stratify -> acak label

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20, #20 derajat
    zoom_range=0.15, #15%
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15, #crop
    horizontal_flip=True,
    fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(number_label, activation="softmax")(headModel) #Ganti 4 Multiclass

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False


#################### WITHOUT PRUNING #######################
pruned_model = 0
# compile our model
if not config.PRUNING:
    print("[INFO] Running Without Prunning...")
    print("[INFO] compiling model...")
    num_images = trainX.shape[0] # * (1 - validation_split)
    opt = Adam(learning_rate=INIT_LR)
    end_step = np.ceil(num_images / BS).astype(np.int32) * EPOCHS
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt,
                  metrics=["accuracy"])

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    print(model.summary())

    # train the head of the network
    print("[INFO] training head...")
    print("[INFO] Total STEP = " + str(end_step))
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS, callbacks=callback) #callbacks=callback
############################################################

################### PRUNING #################################
if config.PRUNING:
    print("[INFO] Running With Prunning...")
    num_images = trainX.shape[0] # * (1 - validation_split)
    end_step = np.ceil(num_images / BS).astype(np.int32) * EPOCHS
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.1,
                                                                 final_sparsity=0.1,
                                                                 begin_step=0.2 * end_step,
                                                                 end_step=end_step),
        # 'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=0.75,
        #                                                           begin_step=500,
        #                                                           end_step=end_step, #1000
        #                                                           frequency=100),
        # 'pruning_policy': tfmot.sparsity.keras.PruneForLatencyOnXNNPack()
    }

    print("[INFO] make a temp directory to debug using tensorboard...")
    log_dir = tempfile.mkdtemp(dir='temp')
    print("Temporary directory:", log_dir)

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3),
        tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
    ] #tf.keras.callbacks.LearningRateScheduler(scheduler)

    def apply_pruning_to_dense(layer): #experimental unused
        if isinstance(layer, Dense):
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer

    model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params) # pruning

    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=opt, metrics=["accuracy"])
    print(model.summary())
    # train the head of the network
    print("[INFO] training head...")
    print("[INFO] Total STEP = " + str(end_step))
    data_amount = 0.5
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS, callbacks=callbacks)
####################################################################

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS) #out [0.0011 0.052 0.2222 0.100 0.25], decode nama lewat lb.classes_

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1) #cek index dari nilai terbesar

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

model_for_export = tfmot.sparsity.keras.strip_pruning(model) #menghilangkan pruning layer


# serialize the model to disk
print("[INFO] saving mask detector model...")
model_for_export.save(config.MODEL_PATH, save_format="h5", include_optimizer=True)

# tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
pruned_tflite_model = converter.convert()

_, pruned_tflite_file = tempfile.mkstemp('.tflite')

with open(pruned_tflite_file, 'wb') as f:
  f.write(pruned_tflite_model)

print('Saved pruned TFLite model to:', pruned_tflite_file)

# if config.PRUNING:
#     def get_gzipped_model_size(file):
#       # Returns size of gzipped model, in bytes.
#       import os
#       import zipfile
#
#       _, zipped_file = tempfile.mkstemp('.zip')
#       with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
#         f.write(file)
#
#       return os.path.getsize(zipped_file)
#
#     print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(config.MODEL_PATH)))

# serialize the label encoder to disk

print("[INFO] saving label encoder...")
f = open(config.ENCODER_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()

print(H.history.keys())


# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)