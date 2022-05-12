import pickle
import numpy as np
from tensorflow.keras import Input, Model, callbacks
from tensorflow.keras.layers import GRU, Dropout, Dense
import matplotlib.pyplot as plt
import matplotlib.style as pltstyle
import argparse 
import utils
from sklearn.preprocessing import LabelBinarizer

IMG_SIZE = 224
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 72
NUM_FEATURES = 2048
EPOCHS = 50

LABELS = set(["pizza", "book", "man", "woman", "dog", "fish", "help", "movie"])

# Define CLI arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--load", required=True,
	help="load pickle data (True), or create pickle data (False)")
ap.add_argument("-n", "--name", required=True,
	help="What to call picture and model created from training")
ap.add_argument("-b", "--batch_size", type=int, default=BATCH_SIZE,
    help="The batch size for the training of the model")
ap.add_argument("-e", "--epochs", type=int, default=EPOCHS,
    help="The amount of epochs for the training of the model")
args = vars(ap.parse_args())

# Check if batch size and epochs CLI argument values are valid
if args["batch_size"] <= 0 or args["epochs"] <= 0:
    print("Please provide a batch size and an epochs amount above 0!")
    exit(0)
else:
    BATCH_SIZE = args["batch_size"]
    EPOCHS = args["epochs"]


print("[INFO] loading feature extractor...")
feature_extractor = utils.build_feature_extractor()

if args["load"] == "True":
    # Load the training and test data and labels from the stored pkl files
    print("[INFO] loading pickle data...")
    with open("./data/traindata.pkl", 'rb') as file:
        train_data = pickle.load(file)

    with open("./data/trainlabels.pkl", 'rb') as file:
        train_labels = pickle.load(file)

elif args["load"] == "False":

    videoPaths = utils.getListOfFiles(".\\video\\dataset\\train\\") #TODO FIX FOR OS
    labels = utils.getListOfLabels(videoPaths)

    # Extract training and test data from videos
    print("[INFO] loading videos...")
    def prepare_all_videos(videoPathList):
        num_videos = len(videoPathList)

        print(f"[INFO] Found {num_videos} video for training...")

        frame_masks = np.zeros(shape=(num_videos, MAX_SEQ_LENGTH), dtype="bool")
        frame_features = np.zeros(
            shape=(num_videos, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )
        # For each video.
        for idx, path in enumerate(videoPathList):
            # Gather all its frames and add to a list.
            frames = utils.load_video(path)
            frames = frames[None, ...]

            temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
            temp_frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
            
            for i, batch in enumerate(frames):
                video_length = batch.shape[0]
                length = min(MAX_SEQ_LENGTH, video_length) #TODO MAYBE DELETE
                for j in range(length):
                    temp_frame_features[i,j,:] = feature_extractor.predict(batch[None, j, :])
                temp_frame_mask[i, :length] = 1
            
            frame_features[idx,] = temp_frame_features.squeeze()
            frame_masks[idx,] = temp_frame_mask.squeeze()


        return (frame_features, frame_masks)

    lb = LabelBinarizer()
    train_labels = lb.fit_transform(labels)

    train_data = prepare_all_videos(videoPaths)
    

    # Save data, to save time later
    with open("./data/traindata.pkl", 'wb') as file:
            pickle.dump(train_data, file)

    with open("./data/trainlabels.pkl", 'wb') as file:
            pickle.dump(train_labels, file)
else:
    print("Specify valid load arguments!")
    exit(0)


print("[INFO] building model...")
frame_features_input = Input((MAX_SEQ_LENGTH, NUM_FEATURES))
mask_input = Input((MAX_SEQ_LENGTH,), dtype="bool")

# Refer to the following tutorial to understand the significance of using `mask`:
# https://keras.io/api/layers/recurrent_layers/gru/

# Create RNN model
x = GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
x = GRU(8)(x)
x = Dropout(0.6)(x)
x = Dense(8, activation="relu")(x)
output = Dense(len(LABELS), activation="softmax")(x)

rnn_model = Model([frame_features_input, mask_input], output)

# Compile the created model
print("[INFO] compiling model...")
rnn_model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

callbacksList = [
    callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=50, min_lr=0.0001
    ),
    callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]

# Train the network
print("[INFO] training head...")
H = rnn_model.fit(
	train_data,
    train_labels,
    validation_split=0.2,
	epochs=EPOCHS,
    batch_size=BATCH_SIZE,
	callbacks=callbacksList)

# Serialize the model to disk
print("[INFO] serializing network...")
rnn_model.save("./models/" + args["name"], save_format="h5")

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