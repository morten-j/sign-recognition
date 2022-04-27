from more_itertools import partition
import cv2
import os
import utils

partitions, labels, classes = utils.get_partitions_and_labels()

# Make folders
for _class in classes:
    if not os.path.exists('./video/image_dataset/' + _class):
        os.makedirs('./video/image_dataset/' + _class)



for sign in partitions["validation"]:
    cam = cv2.VideoCapture("./video/" + sign + ".mp4")
    currentframe = 0
    frameArray = partitions['split'][sign]
    while(True):
        ret,frame = cam.read()
        if ret:
            if frameArray == (1, -1) or currentframe > frameArray[0] and currentframe < frameArray[1]:
                path = './video/image_dataset/' + labels[sign] + '/' + sign +'_frame(' + str(currentframe) + ').jpg'
                cv2.imwrite(path, frame)
            currentframe += 1
        else:
            break
cam.release()
cv2.destroyAllWindows()