import json
import requests
import glob
import os

url = "http://localhost:8080/api/hands"
jsonfile = open("dataset.json")
data = json.load(jsonfile)

video_ids = []

# Make folder for landmarks
if not os.path.exists('./video/landmarks'):
    os.makedirs('./video/landmarks')

# Find all video ids
for i in data:
    for instance in i['instances']:
        video_ids.append(instance['video_id'])

# Send videos to sanic for analysing.
for id in video_ids:
    for file in glob.glob(os.path.join("./video", f'{id}.*')):
        print(f"Sending: {file} to mediapipe")

        files = {
            'video': ('video', open(file, "rb")),
        }
        
        r = requests.post(url, files=files)
        if r.status_code != 200:
            raise Exception(f"Error uploading video! Received status code {r.status_code} with response: {r.text}")

        print(f"Successfully converted to landmarks. Saving in ./video/landmarks/{id}.json")
        landmarkfile = open(f"./video/landmarks/{id}.json", "w")
        landmarkfile.write(r.text)
        print(f"Successfully wrote to file!\n")
        landmarkfile.close()


jsonfile.close()