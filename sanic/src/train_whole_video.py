import sklearn
import pickle
import numpy as np
import pandas as pd
import tensorflow
import keras
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as pltstyle
import argparse
#plt.use("Agg")
import os
from imutils import paths
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import utils
from keras.models import Sequential
from keras.layers import Dense, MaxPooling3D, Conv3D, Flatten, Dropout

LABELS = set(["book", "dog", "fish", "help", "man", "movie", "pizza", "woman"])
IMG_SIZE = 224
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 72

train_data, test_data = utils.get_data_frame_dicts()

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)



print("[INFO] loading videos...")


label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["label"])
)

def prepare_all_videos(df, root_dir):
    frame = []
    video_paths = df["id"].values.tolist()
    labels = df["label"].values
    labels = label_processor(labels[..., None]).numpy()

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add to a list.
        frames = utils.load_video(os.path.join(root_dir, path + ".mp4"))
        frame.append(frames)
        
    return frame, labels

train_data, train_labels = prepare_all_videos(train_df, "video")
test_data, test_labels = prepare_all_videos(test_df, "video")

train_data = np.array(train_data)
test_data = np.array(test_data)


print("LMAOOOOOOO")
print(train_data.shape)
print(train_data[0].shape)


#Akavet
#pkl_filename1 = "traindata.pkl"
#with open(pkl_filename1, 'wb') as file:
#        pickle.dump(train_data, file)

#pkl_filename2 = "trainlabels.pkl"
#with open(pkl_filename2, 'wb') as file:
#        pickle.dump(train_labels, file)

#pkl_filename3 = "testdata.pkl"
#with open(pkl_filename3, 'wb') as file:
#        pickle.dump(test_data, file)

#pkl_filename4 = "testlabels.pkl"
#with open(pkl_filename4, 'wb') as file:
#        pickle.dump(test_labels, file)

class_vocab = label_processor.get_vocabulary()

model = Sequential()
model.add(Conv3D(
            16, (3,3,3), activation='relu', input_shape=(72, 224, 224, 3)
        ))
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(len(class_vocab), activation='softmax')) 
model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "test_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=50, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]


# compile our model (this needs to be done after our setting our
# layers to being non-trainable)
print("[INFO] compiling model...")
#opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / args["epochs"])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
	metrics=["accuracy"])
# train the head of the network for a few epochs (all other layers
# are frozen) -- this will allow the new FC layers to start to become
# initialized with actual "learned" values versus pure random
print("[INFO] training head...")
H = model.fit(
	train_data,
    train_labels,
    validation_split=0.2,
	epochs=50,
	callbacks=callbacks)

# evaluate the network
print("[INFO] evaluating network...")
#predictions = model.predict(x=testX.astype("float32"), batch_size=32)
#print(classification_report(testY.argmax(axis=1),
#	predictions.argmax(axis=1), target_names=lb.classes_))
# plot the training loss and accuracy
_, accuracy = model.evaluate([test_data[0], test_data[1]], test_labels)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(accuracy)


N = 50
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
plt.savefig("test_meme.jpg")


# serialize the model to disk
print("[INFO] serializing network...")
model.save(args["model"], save_format="h5")
# serialize the label binarizer to disk
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()