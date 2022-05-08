import json
import os
import cv2
import numpy as np
import keras
import tensorflow as tf

IMG_SIZE = 224
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 72

pad_frame = np.zeros(shape=(IMG_SIZE,IMG_SIZE,3), dtype="int32")

empty_coor ={
                "x":0.0,
                "y":0.0,
                "z":0.0,
            }
empty_hand = []

for i in range(21):
    empty_hand.append(empty_coor)

empty_frame = dict()
empty_frame["left"] = empty_hand
empty_frame["right"] = empty_hand

def get_partitions_and_labels():
    file_path = 'dataset.json'

    with open(file_path) as ipf:
        content = json.load(ipf)

    classes = []
    labels = dict()
    partition = dict()
    partition['train'] = []
    partition['validation'] = []
    partition['split'] = dict()

    for entry in content:
        gloss = entry['gloss']
        classes.append(gloss)

        for instance in entry['instances']:
            split = instance['split']
            video_id = instance['video_id']
            
            if not os.path.exists("./video/" + video_id + ".mp4"):
                continue

            partition['split'][video_id] = (instance['frame_start'], instance['frame_end'])
            labels[video_id] = gloss

            if split == 'train':
                partition['train'].append(video_id)
            elif split == 'val':
                partition['validation'].append(video_id)
            elif split == 'test':
                partition['validation'].append(video_id)
            else:
                raise ValueError("Invalid split.")

    return partition, labels, classes

def get_data_frame_dicts():
    file_path = 'dataset.json'

    with open(file_path) as ipf:
        content = json.load(ipf)
    
    train_dictionary = {"id": [], "label": []}
    test_dictionary = {"id": [], "label": []}
    for entry in content:
        gloss = entry['gloss']

        for instance in entry['instances']:
            split = instance['split']
            video_id = instance['video_id']
            
            if not os.path.exists("./video/" + video_id + ".mp4"):
                continue

            if split == 'train':
                train_dictionary['id'].append(video_id)
                train_dictionary['label'].append(gloss)
            elif split == 'val':
                train_dictionary['id'].append(video_id)
                train_dictionary['label'].append(gloss)
            elif split == 'test':
                test_dictionary['id'].append(video_id)
                test_dictionary['label'].append(gloss)
            else:
                raise ValueError("Invalid split.")

    return train_dictionary, test_dictionary

def get_data_frame_dicts_mediapipe():
    file_path = 'dataset.json'

    with open(file_path) as ipf:
        content = json.load(ipf)
    
    train_dictionary = {"id": [], "label": [], "landmarks": []}
    test_dictionary = {"id": [], "label": [], "landmarks": []}
    for entry in content:
        gloss = entry['gloss']

        for instance in entry['instances']:
            split = instance['split']
            video_id = instance['video_id']
            video_list = []
            
            if not os.path.exists("./video/landmarks/" + video_id + ".json"):
                continue

            with open("./video/landmarks/" + video_id + ".json") as land:
                landmarks = json.load(land)
            

            if len(landmarks) < 72:
                for frame in range(len(landmarks), 72):
                    landmarks.append(empty_frame)
            if len(landmarks) > 72:
                for frame in range(72, len(landmarks)):
                    landmarks.pop()
    
            for frame in landmarks:
                for mark in frame.values():
                    if len(mark) < 21:
                        for test in range(len(mark), 21):
                            mark.append(empty_coor)
                    if len(mark) > 21:
                        for test2 in range(21, len(mark)):
                            mark.pop()                        

            if len(landmarks) != 72:
                print("dumb")
            for frame_test in landmarks:
                frame_list = []
                if len(frame) != 2:
                    print("blyat")
                for landmark_test in frame_test.values():
                    if len(landmark_test) != 21:
                        print(len(landmark_test))
                        print("NOOOOOO")
                    for coord in landmark_test:
                        if len(coord) != 3:
                            print("LOLOOLOL")
                            print(len(coord))
                        for single_coor in coord.values():
                            frame_list.append(single_coor)
                
                video_list.append(frame_list)
        

            if split == 'train':
                train_dictionary['id'].append(video_id)
                train_dictionary['label'].append(gloss)
                train_dictionary['landmarks'].append(video_list)
            elif split == 'val':
                train_dictionary['id'].append(video_id)
                train_dictionary['label'].append(gloss)
                train_dictionary['landmarks'].append(video_list)
            elif split == 'test':
                test_dictionary['id'].append(video_id)
                test_dictionary['label'].append(gloss)
                test_dictionary['landmarks'].append(video_list)
            else:
                raise ValueError("Invalid split.")

    return train_dictionary, test_dictionary

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret: # Pad data
                for i in range(len(frames), MAX_SEQ_LENGTH):
                    frames.append(pad_frame)
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]] # Converts frame from BGR to RGB
            frames.append(frame)

            if len(frames) == MAX_SEQ_LENGTH:
                break
    finally:
        cap.release()
    frames = frames[:MAX_SEQ_LENGTH]
    return np.array(frames)


def build_feature_extractor():
    feature_extractor = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)

    for layer in feature_extractor.layers:
        layer.trainable = False

    return keras.Model(inputs, outputs, name="feature_extractor")