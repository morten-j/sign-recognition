import numpy as np
import json

filename = "22125"

#with open("./video/landmarks/" + filename + ".json", 'r') as f:
#       json_format = json.load(f)

#json_format = json.load("./video/landmarks/" + filename + ".json")
numpy_format = np.load("./video/numpy_formated_landmarks/" + filename + ".npy", allow_pickle=True)

#json_shape = json_format.shape
numpy_shape = numpy_format.shape

#print("json format shape: " + json_shape)
print(f"numpy format shape: {numpy_shape}. Array Length: {len(numpy_format)}")