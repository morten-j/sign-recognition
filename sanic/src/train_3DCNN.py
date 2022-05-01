from cgi import test
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
from keras.layers import Dense, Conv3D, Dropout, GlobalAveragePooling3D, MaxPool3D, BatchNormalization

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import Session
config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)

IMG_SIZE = 224
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 72
EPOCHS = 50

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--picture", required=True,
	help="What to call picture created from training history")
ap.add_argument("-m", "--model", required=True,
	help="What to call model file")
args = vars(ap.parse_args())

# Get video ids and their labels
train_data, test_data = utils.get_data_frame_dicts()

# Fransform them into Pandas dataframes
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Used to extract how many classes are present in the training data
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["label"])
)
class_vocab = label_processor.get_vocabulary()

def prepare_all_videos(df, root_dir):
    frame = []
    video_paths = df["id"].values.tolist()
    labels = df["label"].values
    labels = label_processor(labels[..., None]).numpy()

    # For each video.
    for idx, path in enumerate(video_paths):
        #print(idx)
        #print(path)
        # Gather all its frames and add to a list.
        frames = utils.load_video(os.path.join(root_dir, path + ".mp4"))
        frame.append(frames)
        
    return frame, labels

train_data, train_labels = prepare_all_videos(train_df, "video")
test_data, test_labels = prepare_all_videos(test_df, "video")

#train_data = np.array(train_data)
#test_data = np.array(test_data)

print(train_labels.shape)


print("[INFO] building model...")
def get_model(frames=None, width=IMG_SIZE, height=IMG_SIZE):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((frames, width, height, 3))

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    #x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    #x = MaxPool3D(pool_size=2)(x)
    #x = BatchNormalization()(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(units=256, activation="relu")(x)
    #x = Dense(units=512, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Dense(len(class_vocab), activation="softmax")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN")
    return model

model = get_model()
model.summary()

print("[INFO] compiling model...")
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=50, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="accuracy", patience=50, verbose=1),
]


print("[INFO] training head...")
H = model.fit(
	train_data,
    train_labels,
    #validation_split=0.2,
	epochs=EPOCHS,
    batch_size=BATCH_SIZE,
	callbacks=callbacks)

# evaluate the network
print("[INFO] evaluating network...")
#predictions = model.predict(x=testX.astype("float32"), batch_size=32)
#print(classification_report(testY.argmax(axis=1),
#	predictions.argmax(axis=1), target_names=lb.classes_))
# plot the training loss and accuracy
_, accuracy = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(accuracy)


N = EPOCHS
pltstyle.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("./pictures/" + args["picture"] + ".jpg")


# serialize the model to disk
print("[INFO] serializing network...")
model.save("./models/" + args["model"], save_format="h5")