import json
import requests
import glob
import os

jsonfile = open("dataset.json")
data = json.load(jsonfile)

video_ids = []

for i in data:
    print(i['gloss'])
    for instance in i['instances']:
        video_ids.append(instance['video_id'])


for id in video_ids:
    for file in glob.glob(os.path.join("./video", f'{id}.*')):
        print(f"Sending: {file} to mediapipe")
        files = {'upload_file': open(file, "rb")}
        r = requests.post("http://localhost:8080/api/hands", files=files)
    
    print(r)
    


jsonfile.close()