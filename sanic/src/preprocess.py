import cv2
import numpy as np

# Given a list of paths to video return a list of these video preprocessed
def prepare_all_videos(videopathList, resize):
    videos = []
    # For each video.
    for idx, path in enumerate(videopathList):
        # Gather all its frames and add to a list.
        video = load_video(path=path, resize=resize)
        videos.append(video)
        
    return np.array(videos)


# Crop video to be smaller and focus on center, since most "action" is there
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

# Load a video and resize and turn it black and white
def load_video(path, resize, convertToBlackAndWhite=True, shouldShow=False, maxLength=72):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)

            if convertToBlackAndWhite:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
               
             # displaying the video
            if shouldShow:
                cv2.imshow("Live", frame)
                cv2.waitKey(30)
            
            frames.append(frame)

            if len(frames) == maxLength:
                break
    finally:
        cap.release()
    return frames