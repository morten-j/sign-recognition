import keras
import numpy as np
import time


class Inception_Classifier:
    #TODO Fix values when done
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, batch_size=64, nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500):
        self.output_directory = output_directory
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1 #hmm
        self.callbacks = None
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.bottleneck_size = 32

        self.model = self.build_model(input_shape, nb_classes)

        #Save data/weigths

    #helper inception module builder
    def build_inception_module(self, input_tensor, stride=1, activation='linear'):
        #check ref
        input_inception = input_tensor

        #Check out
        kernel_size_x = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_x)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_x[i], strides = stride, padding='same', activation=activation, use_bias=False)(input_inception))

        max_pool = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1, padding='same', activation=activation, use_bias=False)(max_pool)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)

        return x

    #build model
    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        #hmm
        x = input_layer
        input_res = input_layer

        for d in range(self.depth):
            x = self.build_inception_module(x)
            #Brug residual?

        gap_level = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_level)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory + 'test_blyat.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model
    #fit
    def fit_test(self, x_train, y_train, x_val, y_val, y_true):
        #x_val and y_val are used to monitor test loss (plots)

        if len(keras.backend.tensorflow_backend._get_available_gpus()) == 0:
            print('no gpu LMAO + ratio')
            #exit()

        #Maybe change
        batch = self.batch_size

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=batch, epochs=self.nb_epochs,verbose=self.verbose, callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'test_blyat_fit.hdf5')

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val, return_df_metrics=False)

        np.save(self.output_directory + 'y_pred.npy', y_pred)

        #convert
        y_pred = np.argmax(y_pred, axis=1)

        #Logging n shit?


    #predict
    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred