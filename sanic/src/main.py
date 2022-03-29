import numpy as np
import keras
import sys

from utils import readucr
from inception_time import Inception_Classifier

#Prepare data
def prepare_data():
    root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

    x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
    x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    num_classes = len(np.unique(y_train))





#Fit classifier

#Create classifier

#Main

#if sys.argv[1] == "Inception":
#    name_classifier = "inception"
#    nb_iter = 5 #huh

    #Setup dataset

root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

num_classes = len(np.unique(y_train))


model = Inception_Classifier(input_shape=x_train.shape[1:], nb_classes=num_classes, output_directory="./")

epochs = 500
batch_size = 32

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)


model = keras.models.load_model("best_model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)
