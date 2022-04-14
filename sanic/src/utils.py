import json


def get_partitions_and_labels():
    file_path = 'dataset'

    with open(file_path) as ipf:
        content = json.load(ipf)

    labels = dict()
    partition = dict()
    partition['train'] = []
    partition['validation'] = []

    for entry in content:
        gloss = entry['gloss']

        for instance in entry['instances']:
            split = instance['split']
            video_id = instance['video_id']

            labels[video_id] = gloss

            if split == 'train':
                partition['train'].append(video_id)
            elif split == 'val':
                partition['validation'].append(video_id)
            elif split == 'test':
                partition['validation'].append(video_id)
            else:
                raise ValueError("Invalid split.")

    return partition, labels
