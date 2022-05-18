from keras import Input, Model
from keras.layers import Dense, Conv3D, Dropout, GlobalAveragePooling3D, MaxPooling3D, BatchNormalization, AveragePooling3D, Flatten


def get_model_big(frames, width, height, depth, classes):

    inputs = Input(shape=(frames, width, height, depth))

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=256, kernel_size=(3,3,3), activation="relu")(x)
    x = Conv3D(filters=512, kernel_size=1 , activation="relu")(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Dense(units=256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    outputs = Dense(units=classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="3DCNN_3")

    return model

def get_the_best_model(frames, width, height, depth, classes):

    inputs = Input(shape=(frames, width, height, depth))

    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(inputs)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = Conv3D(filters=16, kernel_size=3, activation="relu")(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)

    x = Dense(units=128, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)

    outputs = Dense(units=classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="3DCNN_BEST")

    return model


def get_reformed_model(frames, width, height, depth, classes):

    inputs = Input(shape=(frames, width, height, depth))
    x = Conv3D(filters=11, kernel_size=(12,12,12), activation="relu")(inputs)
    x = Conv3D(filters=22, kernel_size=(9,9,9), activation="relu")(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv3D(filters=44, kernel_size=(6,6,6), activation="relu")(x)
    x = Conv3D(filters=88, kernel_size=(3,3,3), activation="relu")(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(units=44, activation="relu")(x)
    x = GlobalAveragePooling3D()(x)
    x = Dropout(0.4)(x)

    x = Dense(units=88, activation="relu")(x)
    x = Dropout(0.5)(x)

    outputs = Dense(units=classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="3DCNN_GOAT")

    return model

def get_yoinked_model(frames, width, height, depth, classes):

    inputs = Input(shape=(frames, width, height, depth))

    x = Conv3D(filters=92, kernel_size=(25,25,6), activation="relu")(inputs)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same', strides=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=216, kernel_size=(15,15,3), activation="relu")(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = Dense(units=2048, activation="relu")(x)
    x = GlobalAveragePooling3D()(x)
    x = Dropout(0.5)(x)

    x = Dense(units=1024, activation="relu")(x)
    x = Dropout(0.5)(x)

    outputs = Dense(units=classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="3DCNN_YOINKED")

    return model

def get_baseline_model(frames, width, height, depth, classes):

    inputs = Input(shape=(frames, width, height, depth))

    x = Conv3D(filters=8, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv3D(filters=16, kernel_size=3, activation="relu")(x)
    x = Conv3D(filters=16, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)    
    x = Dropout(0.3)(x)

    x = Dense(units=64, activation="relu")(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(units=classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="3DCNN_BASELINE")

    return model

def get_experiment_model(frames, width, height, depth, classes):

    inputs = Input(shape=(frames, width, height, depth))

    x = Conv3D(filters=8, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)

    x = Conv3D(filters=16, kernel_size=3, activation="relu")(x)
    x = Conv3D(filters=16, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)

    x = Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)    
    x = Dropout(0.6)(x)

    x = Dense(units=64, activation="relu")(x)
    x = Flatten()(x)
    x = Dropout(0.6)(x)

    outputs = Dense(units=classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="3DCNN_BASELINE")

    return model
