import pickle
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
import matplotlib.style as pltstyle
import argparse 
import os
import utils

IMG_SIZE = 224
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 72
NUM_FEATURES = 2048
EPOCHS = 50


# Get training and test data ids and labels
train_data, test_data = utils.get_data_frame_dicts_mediapipe()

# Setup the training and test ids and labels as a DataFrame (Match the id to the label)
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

print(f"[INFO] Total number of videos for training: {len(train_df)}")
print(f"[INFO] Total number of videos for testing: {len(test_df)}")

# Map the unique labels of the training data to integer indices
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["label"])
)


# Get the vocabulary extracted from the training data previously
class_vocab = label_processor.get_vocabulary()




def prepare_landmarks():
    train_labels = train_df["label"].values
    train_labels = label_processor(train_labels[..., None]).numpy()
    train_data = train_df["landmarks"].values.tolist()
    test = []

    for lol in train_data:
        teste = np.asarray(lol).astype(np.float32)
        test.append(teste)


    return np.array(test), train_labels

lmao, blyat = prepare_landmarks()

print("[INFO] building model...")
feature_input = keras.Input((None, 126))

# Refer to the following tutorial to understand the significance of using `mask`:
# https://keras.io/api/layers/recurrent_layers/gru/

# Create RNN model
x = keras.layers.GRU(16, return_sequences=True)(feature_input)
x = keras.layers.GRU(8)(x)
x = keras.layers.Dropout(0.6)(x)
x = keras.layers.Dense(8, activation="relu")(x)
output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

rnn_model = keras.Model(feature_input, output)

# Compile the created model
print("[INFO] compiling model...")
rnn_model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=50, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]

# Train the network
print("[INFO] training head...")
H = rnn_model.fit(
	lmao,
    blyat,
    validation_split=0.2,
	epochs=EPOCHS,
    batch_size=BATCH_SIZE,
	callbacks=callbacks)

# Evaluate the network
print("[INFO] evaluating network...")
_, accuracy = rnn_model.evaluate( test_data, test_labels)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")

# Plot the training loss and accuracy
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
plt.savefig("./pictures/" + args["name"] + ".jpg")


# Serialize the model to disk
print("[INFO] serializing network...")
rnn_model.save("./models/" + args["name"], save_format="h5")