from asyncio.windows_events import NULL
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

# Get training and test data ids and labels
train_data, test_data = utils.get_data_frame_dicts()

# Setup the training and test ids and labels as a DataFrame (Match the id to the label)
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

print(f"[INFO] Total number of videos for training: {len(train_df)}")
print(f"[INFO] Total number of videos for testing: {len(test_df)}")

# Map the unique labels of the training data to integer indices
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["label"])
)

print("[INFO] loading feature extractor...")
feature_extractor = utils.build_feature_extractor()

if args["load"] == "True":
    # Load the training and test data and labels from the stored pkl files
    print("[INFO] loading pickle data...")
    with open("./data/traindata.pkl", 'rb') as file:
        train_data = pickle.load(file)

    with open("./data/trainlabels.pkl", 'rb') as file:
        train_labels = pickle.load(file)
        
    with open("./data/testdata.pkl", 'rb') as file:
        test_data = pickle.load(file)

    with open("./data/testlabels.pkl", 'rb') as file:
        test_labels = pickle.load(file)

elif args["load"] == "False":
    # Extract training and test data from videos
    print("[INFO] loading videos...")
    def prepare_all_videos(df, root_dir):
        num_videos = len(df)
        video_paths = df["id"].values.tolist()
        labels = df["label"].values
        labels = label_processor(labels[..., None]).numpy()

        frame_masks = np.zeros(shape=(num_videos, MAX_SEQ_LENGTH), dtype="bool")
        frame_features = np.zeros(
            shape=(num_videos, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # For each video.
        for idx, path in enumerate(video_paths):
            # Gather all its frames and add to a list.
            frames = utils.load_video(os.path.join(root_dir, path + ".mp4"))
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


        return (frame_features, frame_masks), labels

    train_data, train_labels = prepare_all_videos(train_df, "video")
    test_data, test_labels = prepare_all_videos(test_df, "video")

    # Save data, to save time later
    with open("./data/traindata.pkl", 'wb') as file:
            pickle.dump(train_data, file)

    with open("./data/trainlabels.pkl", 'wb') as file:
            pickle.dump(train_labels, file)

    with open("./data/testdata.pkl", 'wb') as file:
            pickle.dump(test_data, file)

    with open("./data/testlabels.pkl", 'wb') as file:
            pickle.dump(test_labels, file)
else:
    print("Specify valid load arguments!")
    exit(0)

# Get the vocabulary extracted from the training data previously
class_vocab = label_processor.get_vocabulary()

print("[INFO] building model...")
frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

# Refer to the following tutorial to understand the significance of using `mask`:
# https://keras.io/api/layers/recurrent_layers/gru/

# Create RNN model
x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
x = keras.layers.GRU(8)(x)
x = keras.layers.Dropout(0.6)(x)
x = keras.layers.Dense(8, activation="relu")(x)
output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

rnn_model = keras.Model([frame_features_input, mask_input], output)

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
	train_data,
    train_labels,
    validation_split=0.2,
	epochs=EPOCHS,
    batch_size=BATCH_SIZE,
	callbacks=callbacks)

# Serialize the model to disk
print("[INFO] serializing network...")
rnn_model.save("./models/" + args["name"], save_format="h5")

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