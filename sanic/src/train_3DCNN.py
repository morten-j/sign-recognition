from keras import callbacks
from sklearn.preprocessing import LabelBinarizer
import os
import argparse
import utils
import preprocess
import models
import time
from datetime import timedelta

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,
	help="What to call picture and model created")
ap.add_argument("-m", "--model", required=True,
	help="Which of the models to use")
ap.add_argument("-e", "--epoch", type=int, required=True,
	help="How many epochs in this training?")
ap.add_argument("-b", "--batch", type=int, required=True,
	help="what batch size should be used")
args = vars(ap.parse_args())

IMG_SIZE = 64
BATCH_SIZE = args["batch"]
MAX_SEQ_LENGTH = 72
BLACK_AND_WHITE = 1
RBG = 3
EPOCHS = args["epoch"]
LABELS = set(["book", "dog", "fish", "help", "man", "movie", "pizza", "woman"])


print("[INFO] preparing dataset")
train_videopaths = utils.getListOfFiles(os.path.join("./dataset", "train"))
training_labels = utils.getListOfLabels(train_videopaths, LABELS)

test_videopaths = utils.getListOfFiles(os.path.join("./dataset", "test"))
testing_labels = utils.getListOfLabels(test_videopaths, LABELS)

# Convert labels to: [0,0,0,0,1,0,0,0] format
lb = LabelBinarizer()
train_labels = lb.fit_transform(training_labels)
test_labels = lb.fit_transform(testing_labels)

# Load training and test dataset videos
train_data = preprocess.prepare_all_videos(train_videopaths, resize=(IMG_SIZE, IMG_SIZE))
test_data = preprocess.prepare_all_videos(test_videopaths, resize=(IMG_SIZE, IMG_SIZE))

print(f"Found {len(train_data)} videos for training")
print(f"Found {len(test_data)} videos for testing/evaluation")


print("[INFO] building model...")
# Match/Switch is only available from python 3.10
if args["model"] == 'big':
    model = models.get_model_big(frames=MAX_SEQ_LENGTH, width=IMG_SIZE, height=IMG_SIZE, depth=BLACK_AND_WHITE, classes=len(LABELS))
elif args["model"] == "best":
    model = models.get_the_best_model(frames=MAX_SEQ_LENGTH, width=IMG_SIZE, height=IMG_SIZE, depth=BLACK_AND_WHITE, classes=len(LABELS))
elif args["model"] == "reformed":
    model = models.get_reformed_model(frames=MAX_SEQ_LENGTH, width=IMG_SIZE, height=IMG_SIZE, depth=BLACK_AND_WHITE, classes=len(LABELS))
elif args["model"] == "yoink":
    model = models.get_yoinked_model(frames=MAX_SEQ_LENGTH, width=IMG_SIZE, height=IMG_SIZE, depth=BLACK_AND_WHITE, classes=len(LABELS))
elif args["model"] == "base":
    model = models.get_baseline_model(frames=MAX_SEQ_LENGTH, width=IMG_SIZE, height=IMG_SIZE, depth=BLACK_AND_WHITE, classes=len(LABELS))
elif args["model"] == "exp":
    model = models.get_experiment_model(frames=MAX_SEQ_LENGTH, width=IMG_SIZE, height=IMG_SIZE, depth=BLACK_AND_WHITE, classes=len(LABELS))
else:
    print("[WARNING] no valid model was choosen!")
    exit(0)

#Print summary of used model to CLI
model.summary()

print("[INFO] compiling model...")
model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

callbacksList = [
    # Reduce learning rate if accuracy does not improve in patience epochs
    callbacks.ReduceLROnPlateau(
        monitor="accuracy", factor=0.5, patience=5, min_lr=0.0001
    ),
    # Stop early if model does not improve "val_accuracy" for patience epochs
    callbacks.EarlyStopping(monitor="val_accuracy", patience=25, verbose=1),
]

print("[INFO] training model...")
start_time = time.monotonic()
H = model.fit(
	train_data,
    train_labels,
    validation_split=0.2,
	epochs=EPOCHS,
    batch_size=BATCH_SIZE,
	callbacks=callbacksList
    )
end_time = time.monotonic()
print("[INFO] training took:")
print(timedelta(seconds=end_time - start_time))

# serialize the model to disk
print("[INFO] serializing network...")
model.save("./models/" + args["name"], save_format="h5")

print("[INFO] evaluating network...")
_, accuracy = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")

print("[INFO] plotting training of network...")
# Plot and save the training and validation loss graph
utils.plot_training_acc_loss(H, args["name"])