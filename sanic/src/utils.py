import os
import keras
import pickle
import matplotlib.pyplot as plt
import matplotlib.style as pltstyle

def plot_training_acc_loss(his, name):
    pltstyle.use("ggplot")
    train_acc = his.history["accuracy"]
    train_loss = his.history["loss"]
    epochs = range(1, len(train_acc) + 1)
    plt.figure()
    plt.plot(epochs, train_acc)
    plt.plot(epochs, train_loss)
    plt.title("Training Accuracy and Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend(["train_acc", "train_loss"])
    plt.savefig("./pictures/" + name + ".jpg")
    plt.show()
    


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


def load_model(path):
    if os.path.exists(path):
        return keras.models.load_model(path)
    else:
        return "this is cringe" 

def save_pickle_data(path, objectToSave):
    file = open(path, 'wb')
    pickle.dump(objectToSave, file)

def load_pickle_data(path):
    file = open(path, 'rb')
    return pickle.load(file)