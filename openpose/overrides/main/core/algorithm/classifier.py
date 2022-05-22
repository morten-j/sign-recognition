import numpy as np
import pickle

from main.common.utils import Timer, log
from main.core.algorithm.feature_extractor import FeatureExtractor

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from datetime import timedelta, datetime
import time


class Classifier:
    def __init__(self, model_dir):
        self.feature_extractor = FeatureExtractor()
        self.model = load_model(model_dir)
        self.labels = pickle.load(open(f"{model_dir}/labels.pickle", "rb"))
        self.window_size = pickle.load(open(f"{model_dir}/window_size.pickle", "rb"))
        self.stride = pickle.load(open(f"{model_dir}/stride.pickle", "rb"))
        self.feature_size = pickle.load(open(f"{model_dir}/feature_size.pickle", "rb"))

    def classify(self, input_sequence):
        timer = Timer()

        log(f"Extracting features.")
        timer.start()
        start_time = time.monotonic()
        extracted_feature = self.feature_extractor.extract(input_sequence)  # Extract feature from input
        end_time = time.monotonic()
        append_to_log(f"{datetime.now().strftime('%H:%M:%S')}: Feature Extraction took: {timedelta(seconds=end_time - start_time)}\n")
        log(f"Features extracted in {timer.get_elapsed_seconds()} seconds.")

        # Pad the feature if number of frames less than window size
        if len(extracted_feature) < self.window_size + 1:
            extracted_feature = self._pad_feature(extracted_feature)

        # Create generator based on feature, using same window and stride as model
        feature_generator = TimeseriesGenerator(extracted_feature, np.zeros(extracted_feature.shape),
                                                length=self.window_size, sampling_rate=1, batch_size=1,
                                                stride=self.stride, shuffle=False)

        log(f"Making predictions.")
        timer.start()
        start_time = time.monotonic()
        predictions = self.model.predict_generator(feature_generator)  # Make prediction
        end_time = time.monotonic()
        append_to_log(f"{datetime.now().strftime('%H:%M:%S')}: Classification took: {timedelta(seconds=end_time - start_time)}\n")
        log(f"Predictions made in {timer.get_elapsed_seconds()} seconds.")

        predicted_label_indices = np.argmax(predictions, axis=1)  # Get the indices of predicted labels
        log(f"Predicted label indices: {predicted_label_indices}.")

        predicted_label_index = np.bincount(predicted_label_indices).argmax()  # Choose the most frequent label

        return self.labels[predicted_label_index], predicted_label_index  # Return both label and label index

    def _pad_feature(self, feature):
        difference = self.window_size - len(feature)
        extracted_feature_list = feature.tolist()
        for i in range(difference + 1):
            frame = np.zeros(self.feature_size)
            extracted_feature_list.append(frame)
        return np.array(extracted_feature_list)

def append_to_log(msg: str):
    file_object = open('/sendvideo/video/sendinglog.txt', 'a')
    file_object.write(msg)