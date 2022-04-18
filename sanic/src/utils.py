import json
import os


def get_partitions_and_labels():
    file_path = 'dataset.json'

    with open(file_path) as ipf:
        content = json.load(ipf)

    classes = []
    labels = dict()
    partition = dict()
    partition['train'] = []
    partition['validation'] = []

    for entry in content:
        gloss = entry['gloss']
        classes.append(gloss)

        for instance in entry['instances']:
            split = instance['split']
            video_id = instance['video_id']
            
            if not os.path.exists("./video/" + video_id + ".mp4"):
                continue

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
