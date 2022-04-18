import json
import os
import numpy as np

# Max length/frames of video HARD CODED pt
max_length = 105

frame_array = []
for x in range(42):
    frame_array.append([0, 0, 0])

missing_hand_array = []
for x in range(21):
    missing_hand_array.append([0, 0, 0])

# Make folder for numpy data format
if not os.path.exists('./video/numpy_formated_landmarks'):
    os.makedirs('./video/numpy_formated_landmarks')

for filename in os.listdir("./video/landmarks"):
    with open(os.path.join("./video/landmarks", filename), 'r') as f:
        json_file = json.load(f)
        outer_array = []
        sec_array = []
        for sec in json_file:
            feature_array = []

            for feature in sec.values():
                count= 0
                for coordinates in feature:
                    #print("yassssssss: " + str(count))
                    if count >= 42:
                        print("yas")
                        continue
                    coor_array = []
                    coor_array.append(coordinates["x"])
                    coor_array.append(coordinates["y"])
                    coor_array.append(coordinates["z"])
                    feature_array.append(coor_array)
                    count += 1
                #if len(feature_array) == 21:
                #    for x in range(len(feature_array), 42):
                #        feature_array.append(missing_hand_array)
            if len(feature_array) > 42 and filename == "07076.json":
                print(filename)
                print(feature_array)
                print(len(feature_array))
            sec_array.append(feature_array)
            
        # Pad to fit max
        for x in range(len(sec_array), max_length):
            sec_array.append(frame_array)
        outer_array.append(sec_array)

        np.save(os.path.join("./video/numpy_formated_landmarks", os.path.splitext(filename)[0] + ".npy") ,outer_array)