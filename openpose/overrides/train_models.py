from main.core.algorithm.trainer import LSTMTrainer, TrainingConfiguration
import pickle

DATASET_NAME = '2022_SW_26'
DATASET_DIR = f'/sendvideo/video'

MODELS_DIR = f'{DATASET_DIR}'
TENSORBOARD_DIR = f'{MODELS_DIR}/logs'

MODEL_NAME = "model"
EPOCHS = 25
LAYER_SIZES = [64]
DENSE_LAYERS = [0]
DENSE_ACTIVATION = "relu"
LSTM_LAYERS = [2]
LSTM_ACTIVATION = "tanh"
OUTPUT_ACTIVATION = "softmax"


def train_models():
    dataset = pickle.load(open(f"{DATASET_DIR}/{DATASET_NAME}.pickle", "rb"))

    training_conf = TrainingConfiguration(model_name=MODEL_NAME, epochs=EPOCHS, layer_sizes=LAYER_SIZES,
                                          lstm_activation=LSTM_ACTIVATION, dense_layers=DENSE_LAYERS,
                                          lstm_layers=LSTM_LAYERS, dense_activation=DENSE_ACTIVATION,
                                          output_activation=OUTPUT_ACTIVATION)

    trainer = LSTMTrainer(training_conf)

    models = trainer.create_models(dataset)

    trainer.train_models(models, dataset, TENSORBOARD_DIR)
    trainer.save_models(models, dataset, MODELS_DIR)


if __name__ == '__main__':
    train_models()