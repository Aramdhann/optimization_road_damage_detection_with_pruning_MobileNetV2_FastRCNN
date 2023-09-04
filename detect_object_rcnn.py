# USAGE
# import the necessary packages
import tensorflow_model_optimization as tfmot
from pyimagesearch.nms import non_max_suppression
from pyimagesearch import config
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
from pyimagesearch import config
import pandas as pd

# =================
# PENS 2023
# =================

# imagePath = 'images/63.jpg'
# imagePath = 'images/road_01.jpg'
# imagePath = 'images/375.jpg'
# imagePath = 'images/road_03.jpg'
# imagePath = 'images/171.jpg'
imagePath = 'images/road_04.jpg'

# imagePath = 'images/road_07.jpg'
# imagePath = 'images/road_06.jpg'
# imagePath = 'images/429.jpg'
# imagePath = 'images/road_10.png'


# load the our fine-tuned model and label encoding from disk
print("[INFO] loading model and label encoding...")

model = load_model(config.MODEL_PATH)

lb = pickle.loads(open(config.ENCODER_PATH, "rb").read())

print(model.summary())
print(lb)
cv2.waitKey(0)

# load the input image from disk
image = cv2.imread(imagePath)
image = imutils.resize(image, width=500)

# run selective search on the image to generate bounding box proposal
# regions
print("[INFO] running selective search...")
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
rects = ss.process() #proposed bounding box

# initialize the list of region proposals that we'll be classifying
# along with their associated bounding boxes
proposals = []
boxes = []

# loop over the region proposal bounding box coordinates generated by
# running selective search
for (x, y, w, h) in rects[:config.MAX_PROPOSALS_INFER]:
    # extract the region from the input image, convert it from BGR to
    # RGB channel ordering, and then resize it to the required input
    # dimensions of our trained CNN
    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, config.INPUT_DIMS,
                     interpolation=cv2.INTER_CUBIC)

    # further preprocess by the ROI
    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    # update our proposals and bounding boxes lists
    proposals.append(roi)
    boxes.append((x, y, x + w, y + h))

# convert the proposals and bounding boxes into NumPy arrays
proposals = np.array(proposals, dtype="float32")
boxes = np.array(boxes, dtype="int32")
print("[INFO] proposal shape: {}".format(proposals.shape))

# classify each of the proposal ROIs using fine-tuned model
print("[INFO] classifying...")
proba = model.predict(proposals)

# find the index of all predictions that are positive for the
# multiclass
print("[INFO] applying NMS...")
labels = lb.classes_[np.argmax(proba, axis=1)] #[00, L00, R00, ....,]

# Create dataframe
data = []
for idx in range(len(labels)):
    data.append([idx, boxes[idx], proba[idx], np.max(proba[idx]), labels[idx]])
data_df = pd.DataFrame(data, columns=["idx", "box", "scores", "score", "label"])

# Create dictionary to save labelled box and labelled score
unique_label = data_df["label"].drop_duplicates().to_list()
data_dict = {}
for label in unique_label:
    data_label = data_df[data_df["label"] == label][["box", "score"]]
    data_label = data_label[data_label["score"] >= config.MIN_PROBA]
    data_dict[label] = {
        "box": data_label["box"].to_list(),
        "score": data_label["score"].to_list(),
    }

#Fungsi text
def draw_text(img, text,
          # font=cv2.FONT_HERSHEY_PLAIN,
          font=cv2.FONT_HERSHEY_SIMPLEX,
          pos=(0, 0),
          font_scale=0.8,
          font_thickness=2,
          text_color=(0, 0, 0),
          text_color_bg=(0, 255, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    rect_w = text_w + 6  # Adjust the rectangle width according to the text size
    rect_h = text_h + 10  # Adjust the rectangle height according to the text size
    cv2.rectangle(img, pos, (x + rect_w, y - rect_h), text_color_bg, -1)
    # cv2.rectangle(img, pos, (x + 5 + text_w + 6, y - text_h - 8), text_color_bg, -1)
    cv2.putText(img, text, (x + 5, y - 12 + int(font_scale * 10)), font, font_scale, text_color, font_thickness)

    return text_size

# draw selective search
# for proba in rects:
    # for label in unique_label:
    #     idx_suppression = non_max_suppression(np.array(data_dict[label]["box"]), np.array(data_dict[label]["score"]))
    #     for idx in idx_suppression:
    #         if label != "no_label":
    #             x, y, w, h = proba
    #             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #             cv2.imshow('Selective Search', image)

# join the overlapping bounding box + draw
data_dict_suppression = {}
for label in unique_label:
    idx_suppression = non_max_suppression(np.array(data_dict[label]["box"]), np.array(data_dict[label]["score"]))
    for idx in idx_suppression:
        if label != "no_label":
            if data_dict[label]["box"] and data_dict[label]["score"]:
                (startX, startY, endX, endY) = data_dict[label]["box"][idx]
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "{} {:.2f}%".format(label, data_dict[label]["score"][idx] * 100)
                # Add text
                # cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 0), 2)
                draw_text(image, text=text, pos=(startX,startY))

# show the output image *after* running NMS
cv2.imshow("Crack Detector", image)
cv2.waitKey(0)
