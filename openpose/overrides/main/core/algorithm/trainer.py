import pickle

from main.common.utils import Timer, log
from main.core.algorithm.lstm_model import LSTMModel

from tensorflow.keras.callbacks import TensorBoard

from datetime import datetime


class TrainingConfiguration:
    def __init__(self, model_name="model", epochs=25, dense_layers=None, layer_sizes=None, lstm_layers=None,
                 lstm_activation="tanh", dense_activation="relu", output_activation="softmax", dropout=0.25):
        self.model_name = model_name
        self.epochs = epochs
        self.dense_layers = dense_layers
        self.layer_sizes = layer_sizes
        self.lstm_layers = lstm_layers
        self.lstm_activation = lstm_activation
        self.dense_activation = dense_activation
        self.output_activation = output_activation
        self.dropout = dropout
        if lstm_layers is None:
            self.lstm_layers = [2]
        if layer_sizes is None:
            self.layer_sizes = [64]
        if dense_layers is None:
            self.dense_layers = [0]
        self.combinations = len(self.dense_layers) * len(self.layer_sizes) * len(self.lstm_layers)


class LSTMTrainer:
    def __init__(self, conf):
        self.conf = conf

    def create_models(self, dataset):
        log("Creating models.")
        models = []

        output_size = len(dataset.labels)  # Set number of output neurons to number of labels

        # Create a model for each possible combination based on hyper parameters from trainer configuration
        for layer_size in self.conf.layer_sizes:
            for lstm_layer in self.conf.lstm_layers:
                for dense_layer in self.conf.dense_layers:
                    model_name = f'{self.conf.model_name}_{lstm_layer}_lstm-{layer_size}_units-' \
                                 f'{dense_layer}_dense-{datetime.now().strftime("%d_%m_%Y-%H_%M_%S")}'

                    model = LSTMModel(name=model_name, lstm_layers=lstm_layer, dense_layers=dense_layer,
                                      layer_size=layer_size, input_shape=(dataset.window, dataset.feature_size),
                                      output_size=output_size, lstm_activation=self.conf.lstm_activation,
                                      dense_activation=self.conf.dense_activation,
                                      output_activation=self.conf.output_activation, dropout=self.conf.dropout)

                    models.append(model)

        log("Models created.")
        return models

    def train_models(self, models, dataset, tensorboard_dir=""):
        log("Training models.")

        for i, model in enumerate(models):

            log(f'Training "{model.name}".')

            timer = Timer()
            timer.start()

            if tensorboard_dir is not "":
                tensorboard = TensorBoard(log_dir=f'{tensorboard_dir}/{model.name}')

                model.src.fit_generator(dataset.train, epochs=self.conf.epochs, validation_data=dataset.val,
                                        callbacks=[tensorboard])
            else:
                model.src.fit_generator(dataset.train, epochs=self.conf.epochs, validation_data=dataset.val)

            test_evaluation = model.src.evaluate_generator(dataset.test)  # Evaluate model on test set

            log(f'Training for "{model.name}" complete. Elapsed time: {timer.get_elapsed_time()}')
            log(f'{i + 1} out of {self.conf.combinations} models trained.')
            log(f'Test loss, Test accuracy: {test_evaluation}')

    def save_models(self, models, dataset, save_dir):
        log("Saving models.")

        for model in models:
            model.src.save(f'{save_dir}/{model.name}.model')

            pickle_out = open(f'{save_dir}/{model.name}.model/labels.pickle', 'wb')
            pickle.dump(dataset.labels, pickle_out)
            pickle_out.close()

            pickle_out = open(f'{save_dir}/{model.name}.model/window_size.pickle', 'wb')
            pickle.dump(dataset.window, pickle_out)
            pickle_out.close()

            pickle_out = open(f'{save_dir}/{model.name}.model/stride.pickle', 'wb')
            pickle.dump(dataset.stride, pickle_out)
            pickle_out.close()

            pickle_out = open(f'{save_dir}/{model.name}.model/feature_size.pickle', 'wb')
            pickle.dump(dataset.feature_size, pickle_out)
            pickle_out.close()

            log(f'"{model.name}" model saved.')

        log("Models saved.")
