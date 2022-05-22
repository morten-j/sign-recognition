import pickle
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import confusion_matrix as conf_mat
from matplotlib import cm

DATASET_NAME = '2022_SW_26'
MODEL_NAME = "sw_26_model"
DATASET_DIR = f'/sendvideo/video/{DATASET_NAME}.pickle'
#MODEL_DIR = f"./main/core/algorithm/models/{DATASET_NAME}/{MODEL_NAME}"

MODEL_DIR = f'../sendvideo/video/models/sw_26_model_2_lstm-64_units-0_dense-22_05_2022-07_18_14.model'

def get_ground_truth(generator):
    ground_truth = []
    index = 0
    counting = True
    while counting:
        if index + generator.length < len(generator.targets):
            target = generator.targets[index:index+generator.length]
            ground_truth.append(np.bincount(target).argmax())
            index += generator.stride
        else:
            counting = False
    return ground_truth


def get_predictions_and_ground_truth():
    model = load_model(MODEL_DIR)

    dataset = pickle.load(open(DATASET_DIR, 'rb'))

    test_generator = TimeseriesGenerator(dataset.test.data, dataset.test.targets,
                                         length=dataset.window, sampling_rate=1, batch_size=1,
                                         stride=dataset.stride, shuffle=False, start_index=0)

    predictions = model.predict_generator(test_generator)

    return np.argmax(predictions, axis=1), get_ground_truth(test_generator)


def normalize_confusion_matrix(confusion_matrix):
    normalized = []
    for i, row in enumerate(confusion_matrix):
        sum = np.sum(row)
        new_row = []
        for j, x in enumerate(row):
            new_row.append(x / sum)
        normalized.append(new_row)
    return np.array(normalized)


def generate_confusion_matrix():
    predictions, ground_truth = get_predictions_and_ground_truth()

    confusion_matrix = conf_mat(ground_truth, predictions)
    confusion_matrix_normalized = normalize_confusion_matrix(confusion_matrix)

    labels = pickle.load(open(f'{MODEL_DIR}/labels.pickle', 'rb'))

    df_confusion_matrix = pd.DataFrame(confusion_matrix, index=[i for i in labels], columns=[i for i in labels])
    df_confusion_matrix_normalized = pd.DataFrame(confusion_matrix_normalized, index=[i for i in labels], columns=[i for i in labels])

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_confusion_matrix, annot=True, cmap=cm.get_cmap('Blues'), fmt='.3g')

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_confusion_matrix_normalized, annot=True, cmap=cm.get_cmap('Blues'), fmt='.3g')

    plt.savefig('confmat.png')


if __name__ == '__main__':
    generate_confusion_matrix()