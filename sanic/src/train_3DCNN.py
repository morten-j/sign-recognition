from cgi import test
from multiprocessing import pool
import pickle
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
import matplotlib.style as pltstyle
import argparse
import os
import utils
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv3D, Dropout, GlobalAveragePooling3D, MaxPooling3D, BatchNormalization, AveragePooling3D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer

# TODO fix import, så de kommer fra det samme sted (keras og tensorflow)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,
	help="What to call picture and model created")
ap.add_argument("-m", "--model", required=True,
	help="Which of the models to use")
ap.add_argument("-e", "--epoch", required=True,
	help="How many epochs in this training?")
ap.add_argument("-b", "--batch", required=True,
	help="what batch size should be used")
args = vars(ap.parse_args())

IMG_SIZE = 64
BATCH_SIZE = int(args["batch"])
MAX_SEQ_LENGTH = 72
EPOCHS = int(args["epoch"])
LABELS = set(["pizza", "book", "man", "woman", "dog", "fish", "help", "movie"])


print("[INFO] preparing dataset")

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        elif entry.lower().endswith(".mp4"):
            allFiles.append(fullPath)
                
    return allFiles

def getListOfLabels(videoPathList):
    allLabels = list()

    for path in videoPathList:
        label = path.split(os.path.sep)[-2]
        if label not in LABELS:
            continue
        allLabels.append(label)

    return allLabels

videopaths = getListOfFiles("./videos")
labels = getListOfLabels(videopaths)

# Used to extract how many classes are present in the training data
#label_processor = keras.layers.StringLookup(
#    num_oov_indices=0, vocabulary=np.unique(labels)
#)
#class_vocab = label_processor.get_vocabulary()

def prepare_all_videos(videopathList):
    videos = []

    # For each video.
    for idx, path in enumerate(videopathList):
        # Gather all its frames and add to a list.
        video = utils.load_video(path=path, resize=(IMG_SIZE, IMG_SIZE), convertToBlackAndWhite=True, shouldShow=False)
        videos.append(video)
        
    return videos

lb = LabelBinarizer()
train_labels = lb.fit_transform(labels)

train_data = prepare_all_videos(videopaths)

print(np.shape(train_data))

print(f"Found {len(train_data)} videos")

# Maybe useful
#fixed_labels = to_categorical(train_labels, len(class_vocab))

# Maybe UseFull
#train_data = np.array(train_data)
#test_data = np.array(test_data)


print("[INFO] building model...")
def get_model(frames=None, width=IMG_SIZE, height=IMG_SIZE):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((frames, width, height, 3))

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    #x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    #x = MaxPool3D(pool_size=2)(x)
    #x = BatchNormalization()(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(units=256, activation="relu")(x)
    #x = Dense(units=512, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(units=len(LABELS), activation="softmax")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN")
    return model

def get_model2(frames=None, width=IMG_SIZE, height=IMG_SIZE):

    inputs = keras.Input((frames, width, height, 3))

    x = Conv3D(filters=96, kernel_size=3, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = AveragePooling3D(pool_size=(2,2,2))(x)

    x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = BatchNormalization()(x)
    x = AveragePooling3D(pool_size=(2,2,2))(x)

    x = Conv3D(filters=384, kernel_size=3, activation="relu")(x)
    x = Conv3D(filters=384, kernel_size=3, activation="relu")(x)
    x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)

    x = GlobalAveragePooling3D()(x)

    x = Dense(units=256, activation="relu")(x)
    outputs = Dense(len(LABELS), activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="3DCNN_2")
    return model

def get_model3(frames=MAX_SEQ_LENGTH, width=IMG_SIZE, height=IMG_SIZE):

    inputs = keras.Input((frames, width, height, 3))

    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(inputs)
    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=256, kernel_size=(3,3,3), activation="relu")(x)
    x = Conv3D(filters=512, kernel_size=1 , activation="relu")(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Dense(units=256, activation="relu")(x)
    x = Flatten()(x)
    outputs = Dense(units=len(LABELS), activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="3DCNN_3")


    return model

def get_the_best_model(frames=MAX_SEQ_LENGTH, width=IMG_SIZE, height=IMG_SIZE, depth=1):

    inputs = keras.Input(shape=(frames, width, height, depth))

    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(inputs)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = Conv3D(filters=16, kernel_size=3, activation="relu")(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Dense(units=128, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)

    outputs = Dense(units=len(LABELS), activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="3DCNN_BEST")

    return model

# Match/Switch is only available from python 3.10
if args["model"] == '1':
    model = get_model()
elif args["model"] == '2':
    model = get_model2()
elif args["model"] == '3':
    model = get_model3()
elif args["model"] == "best":
    model = get_the_best_model()
else:
    print("[WARNING] no valid model was choosen!")
    exit(0)

model.summary()

print("[INFO] compiling model...")
model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=50, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=50, verbose=1),
]


print("[INFO] training model...")
H = model.fit(
	train_data,
    train_labels,
    validation_split=0.2,
	epochs=EPOCHS,
    batch_size=BATCH_SIZE,
	callbacks=callbacks)


# serialize the model to disk
print("[INFO] serializing network...")
model.save("./models/" + args["name"], save_format="h5")


# evaluate the network
#print("[INFO] evaluating network...")
#predictions = model.predict(x=testX.astype("float32"), batch_size=32)
#print(classification_report(testY.argmax(axis=1),
#	predictions.argmax(axis=1), target_names=lb.classes_))
# plot the training loss and accuracy
#_, accuracy = model.evaluate(test_data, test_labels)
#print(f"Test accuracy: {round(accuracy * 100, 2)}%")
#print(accuracy)


#N = EPOCHS
#pltstyle.use("ggplot")
#plt.figure()
#plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
#plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
#plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
#plt.title("Training Loss and Accuracy on Dataset")
#plt.xlabel("Epoch #")
#plt.ylabel("Loss/Accuracy")
#plt.legend(loc="lower left")
#plt.savefig("./pictures/" + args["name"] + ".jpg")