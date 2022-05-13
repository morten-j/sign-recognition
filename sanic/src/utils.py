import os
import tensorflow
import matplotlib.pyplot as plt
import matplotlib.style as pltstyle
import numpy as np

IMG_SIZE = 64
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 72


def plot_training(his, metric, name):
    pltstyle.use("ggplot")
    train_metrics = his.history[metric]
    val_metrics = his.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.figure()
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()
    plt.savefig("./pictures/" + name + ".jpg")


def getListOfFiles(dirName):
    # create a list of file and sub directories 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        # Create full path for video
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        elif entry.lower().endswith(".mp4"):
            allFiles.append(fullPath)
                
    return allFiles

def getListOfLabels(videoPathList, labels):
    allLabels = list()

    for path in videoPathList:
        # Extract name of folder (which is the label)
        label = path.split(os.path.sep)[-2]
        if label not in labels:
            continue
        allLabels.append(label)

    return allLabels


def build_feature_extractor(shape=(IMG_SIZE, IMG_SIZE, 3)):
    feature_extractor = tensorflow.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(shape),
    )

    preprocess_input = tensorflow.keras.applications.inception_v3.preprocess_input

    inputs = tensorflow.keras.Input(shape)
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)

    for layer in feature_extractor.layers:
        layer.trainable = False

    return tensorflow.keras.Model(inputs, outputs, name="feature_extractor")


def load_model(path):
    if os.path.exists(path):
        return tensorflow.keras.models.load_model(path)
    else:
        return "this is cringe" 
