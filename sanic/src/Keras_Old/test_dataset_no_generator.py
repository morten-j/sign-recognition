import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import utils
import sanic.src.Keras_Old.data_generator as data_generator
import tensorflow as tf
import json
import os

partition, labels, classes = utils.get_partitions_and_labels()
data = []


#for file in partition["train"]:
#    with open(os.path.join("./video/landmarks/",file + ".json"), 'r') as f:
#        data.append(json.load(f))

#data.append(np.load('video/numpy_formated_landmarks/' + "07068" + '.npy', allow_pickle=True))

#test = tf.convert_to_tensor(data[0][0])

#dataset = tf.data.Dataset.from_tensor_slices(data[0][0])







for file in partition["train"]:
    data.append(np.load('video/numpy_formated_landmarks/' + file + '.npy', allow_pickle=True))

test = []

for sign in data:
#    print(len(sign))
    test.append(tf.ragged.constant(sign))
#    for frame in sign:
#        print(len(frame))
#        for landmarks in frame:
#            print(len(landmarks))
#            for coor in landmarks:
#                print(len(coor))


#data.append(np.load('video/numpy_formated_landmarks/' + "07068" + '.npy', allow_pickle=True))
yo = []
for blyat in test:
    temp = blyat.to_tensor()
    temp2 = tf.linalg.normalize(temp, axis = None)
    temp3 = tf.convert_to_tensor(temp2[0])
    #print(temp3)
    #temp4 = tf.RaggedTensor.from_tensor(temp3, padding=0.0)
    #print("lmao")
    yo.append(temp3)


dataset = tf.data.Dataset.from_tensor_slices(yo)

# Find number of classes
num_classes = len(np.unique(classes))

# Make model
def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=(42, 3))
#keras.utils.plot_model(model, show_shapes=True)



# Train the model
epochs = 250
batch_size = 124

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "test_model.h5", save_best_only=True, monitor="loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=50, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=['sparse_categorical_accuracy'],
)
history = model.fit(
    x=dataset,
    #y=validation_generator,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    #validation_split=0.2,
    verbose=1,
    use_multiprocessing=True,
    workers=6,
)