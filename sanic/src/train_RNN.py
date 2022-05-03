import pickle
import numpy as np
import pandas as pd
from tensorflow import keras
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

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--load", required=True,
	help="load pickle data, or create pickle data")
ap.add_argument("-p", "--picture", required=True,
	help="What to call picture created from training history")
ap.add_argument("-m", "--model", required=True,
	help="What to call model file")
args = vars(ap.parse_args())


train_data, test_data = utils.get_data_frame_dicts()

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["label"])
)


print("[INFO] loading feature extractor...")
feature_extractor = utils.build_feature_extractor()


if args["load"] == "True":

    with open("./data/traindata.pkl", 'rb') as file:
        train_data = pickle.load(file)

    with open("./data/trainlabels.pkl", 'rb') as file:
        train_labels = pickle.load(file)
        
    with open("./data/testdata.pkl", 'rb') as file:
        test_data = pickle.load(file)

    with open("./data/testlabels.pkl", 'rb') as file:
        test_labels = pickle.load(file)

elif args["load"] == "False":
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
    print("Specify valid load argument!!!")
    exit(0)


class_vocab = label_processor.get_vocabulary()


print("[INFO] building model...")
frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

# Refer to the following tutorial to understand the significance of using `mask`:
# https://keras.io/api/layers/recurrent_layers/gru/
x = keras.layers.GRU(524, return_sequences=True)(
    frame_features_input, mask=mask_input
)
x = keras.layers.GRU(124)(x)
x = keras.layers.Dropout(0.4)(x)
x = keras.layers.Dense(64, activation="relu")(x)
output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

rnn_model = keras.Model([frame_features_input, mask_input], output)


print("[INFO] compiling model...")
rnn_model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=50, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]


print("[INFO] training head...")
H = rnn_model.fit(
	train_data,
    train_labels,
    validation_split=0.2,
	epochs=EPOCHS,
    batch_size=BATCH_SIZE,
	callbacks=callbacks)

# evaluate the network
print("[INFO] evaluating network...")
#predictions = model.predict(x=testX.astype("float32"), batch_size=32)
#print(classification_report(testY.argmax(axis=1),
#	predictions.argmax(axis=1), target_names=lb.classes_))
# plot the training loss and accuracy
_, accuracy = rnn_model.evaluate([test_data[0], test_data[1]], test_labels)
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
rnn_model.save("./models/" + args["model"], save_format="h5")