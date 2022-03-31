import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

# Python Generator that yields an image every call
def getImagesFromVideo(video):
    videoWrapper = cv2.VideoCapture(video)

    if videoWrapper.isOpened() == False:
        raise Exception("Could not open video")
    
    while videoWrapper.isOpened():
        wasReadSuccess, image = videoWrapper.read()

        if wasReadSuccess == False:
            break # End once video is done reading.

        yield image
        
    videoWrapper.release()

def getLandmarksFromVideo(video, flipped=False):
    results = []
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as handtracker:
        
        
        for image in getImagesFromVideo(video):
            if flipped:
                image = cv2.flip(image, 1)

            result = handtracker.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            hands = {'left': [], 'right': []}
            for index, hand_landmarks in enumerate(result.multi_hand_landmarks):
                handedness = 'left' if index == 0 else "right"

                for landmark in hand_landmarks.landmark:
                    hands[handedness].append({'x': landmark.x, 'y': landmark.y, 'z': landmark.z})
                
            results.append(hands)

        print(f'Frame 1, left-hand, wrist: {results[0]["left"][0]}')
        return results