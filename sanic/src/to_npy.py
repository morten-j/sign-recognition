import json
import os
import numpy as np

# Make folder for numpy data format
if not os.path.exists('./video/numpy_formated_landmarks'):
    os.makedirs('./video/numpy_formated_landmarks')

for filename in os.listdir("./video/landmarks"):
   with open(os.path.join("./video/landmarks", filename), 'r') as f:
       json_file = json.load(f)
       np.save(os.path.join("./video/numpy_formated_landmarks", os.path.splitext(filename)[0] + ".npy") ,json_file)