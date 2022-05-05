import pickle
import argparse
import utils
from tensorflow import keras

test_data = utils.get_data_frame_dicts()

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="model file to load")
args = vars(ap.parse_args())


with open("./data/testdata.pkl", 'rb') as file:
    test_data = pickle.load(file)

with open("./data/testlabels.pkl", 'rb') as file:
    test_labels = pickle.load(file)

print("[INFO] Loading model...")
rnn_model = keras.models.load_model("models/" + args["model"])

print("[INFO] evaluating network...")
prediction = rnn_model.predict(test_data)
_, accuracy = rnn_model.evaluate(test_data, test_labels)
print(f"Test accuracy: {round(accuracy * 100, 2)}%")