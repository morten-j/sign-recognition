import json
import os
import cv2
import numpy as np

IMG_SIZE = 224
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 72

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

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    frames = frames[:MAX_SEQ_LENGTH]
    return np.array(frames)


def prepare_videos(df, root_dir):
    frame = []
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    labels = df["tag"].values
    labels = label_processor(labels[..., None]).numpy()

    # For each video.
    for idx, path in enumerate(video_paths):
        #print(idx)
        #print(path)
        # Gather all its frames and add to a list.
        frames = load_video(os.path.join(root_dir, path))
        frame.append(frames)
        
    return frame, labels