import base64
import json
import cv2
import requests
from requests import Response

# Converts a base64 bytes file to its blob string representation
def convert_base64_to_string_format(b64, type):
    b64_string = str(b64)
    return f'data:{type};base64,{b64_string[2:len(b64_string)-1]}'

def extract_frames_from_video(video_path):
    sequence = []

    # Create OpenCV capture using video as src
    capture = cv2.VideoCapture(video_path)
    # Check if capture opened successfully
    if (capture.isOpened() == False):
        raise Exception("Error opening video file.")
    # Read frames until video is completed
    while capture.isOpened():
        # Capture every frame
        ret, frame = capture.read()
        if ret:
            sequence.append(frame)

        # Break the loop when no more frames
        else:
            break
    # When all frames are extracted, release the video capture object
    capture.release()
    # Close all the OpenCV window frames
    cv2.destroyAllWindows()

    return sequence



frames = extract_frames_from_video('./video/video.webm')

valid_base64_data = []
for frame in frames:
    ret, buffer = cv2.imencode('.jpg', frame)
    b64 = base64.b64encode(buffer)
    valid_base64_data.append(convert_base64_to_string_format(b64, 'image/jpg'))



# sending get request and saving the response as response object
r: Response = requests.post(url="http://localhost:5000/recognize", data=json.dumps(valid_base64_data))

print(f"response: {r.status_code} {r.reason} {r.text}")
# extracting data in json format
