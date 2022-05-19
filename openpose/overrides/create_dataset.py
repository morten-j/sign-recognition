from main.core.algorithm.dataset import DataSetCreator
import pickle

CREATE_RAW_DATA = True

DATASET_NAME = '2022_SW_26'
DATASET_DIR = f'/sendvideo/video'

WINDOW_LENGTH = 60
STRIDE = 5
BATCH_SIZE = 512
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1


def create_raw_data():
    dataset_creator = DataSetCreator()

    raw_dataset, labels = dataset_creator.create_raw_dataset_from_src(f'{DATASET_DIR}/src')

    dataset_creator.save_raw_dataset(raw_dataset, DATASET_NAME, labels, DATASET_DIR)


def create_dataset():
    dataset_creator = DataSetCreator()

    raw_dataset = pickle.load(
        open(f"{DATASET_DIR}/{DATASET_NAME}_raw.pickle", "rb"))
    labels = pickle.load(open(f"{DATASET_DIR}/{DATASET_NAME}_labels.pickle", "rb"))

    dataset = dataset_creator.create_dataset(raw_dataset, labels, window_length=WINDOW_LENGTH, stride=STRIDE,
                                             batch_size=BATCH_SIZE, val_split=VAL_SPLIT, test_split=TEST_SPLIT)
    dataset.save(DATASET_NAME, DATASET_DIR)


if __name__ == '__main__':
    if CREATE_RAW_DATA:
        create_raw_data()
    create_dataset()

