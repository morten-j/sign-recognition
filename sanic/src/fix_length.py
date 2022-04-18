import os
import numpy as np

# Find max length
max_length = 0

for filename in os.listdir("./video/numpy_formated_landmarks"):
    temp = np.load('video/numpy_formated_landmarks/' + os.path.splitext(filename)[0] + '.npy', allow_pickle=True)
    for frames in temp:
        if len(frames) > max_length:
            max_length = len(frames)

frame_array = []
for x in range(42):
    frame_array.append([0, 0, 0])

# Pad arrays
for filename in os.listdir("./video/numpy_formated_landmarks"):
    temp = np.load('video/numpy_formated_landmarks/' + os.path.splitext(filename)[0] + '.npy', allow_pickle=True)
    for frames in temp:
        if len(frames) < max_length:
            for missing in range(len(frames), max_length):
                blyat = np.append(frames, frame_array)
        
    print(len(frames))






