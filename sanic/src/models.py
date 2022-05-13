from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv3D, Dropout, GlobalAveragePooling3D, MaxPooling3D, BatchNormalization, AveragePooling3D, Flatten



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
    x = Dropout(0.7)(x)
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