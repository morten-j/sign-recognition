import tensorflow as tf
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import CSVLogger

def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")


classes = np.unique(np.concatenate((y_train, y_test), axis=0))

plt.figure()
for c in classes:
    c_x_train = x_train[y_train == c]
    plt.plot(c_x_train[0], label="class " + str(c))
plt.legend(loc="best")
plt.show()
plt.close()


x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))


num_classes = len(np.unique(y_train))
depth = 6


idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]


y_train[y_train == -1] = 0
y_test[y_test == -1] = 0
kernel_size=41
nb_filters=32

def build_inception_module(input_tensor, stride=1, activation='linear'):
    #check ref
    input_inception = input_tensor

    #Check out
    kernel_size_x = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_x)):
        conv_list.append(keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_x[i], strides = stride, padding='same', activation=activation, use_bias=False)(input_inception))

    max_pool = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = keras.layers.Conv1D(filters=nb_filters, kernel_size=1, padding='same', activation=activation, use_bias=False)(max_pool)

    conv_list.append(conv_6)

    x = keras.layers.Concatenate(axis=2)(conv_list)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)

    return x


def make_model(input_shape, nb_classes):
    input_layer = keras.layers.Input(input_shape)

    x = input_layer

    for d in range(depth):
        x = build_inception_module(x)
        #Brug residual?

    gap_level = keras.layers.GlobalAveragePooling1D()(x)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_level)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=x_train.shape[1:], nb_classes=num_classes)
#keras.utils.plot_model(model, show_shapes=True)




epochs = 100
batch_size = 32

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "test_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=50, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    keras.callbacks.CSVLogger("training.log", separator=",", append=True),
]
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=['sparse_categorical_accuracy'],
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

model.save("test_model")

model = keras.models.load_model("best_model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)



metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()

