import sys
import os

from sys import platform

from .paths import PYTHON_PATH, PYTHON_RELEASE_PATH, X64_RELEASE_PATH, BIN_PATH, MODELS_PATH



# Import Openpose (Windows/Ubuntu/OSX)
try:
    # Windows Import
    if platform == "win32":
        sys.path.append(PYTHON_RELEASE_PATH)
        os.environ['PATH'] = os.environ['PATH'] + ';' + X64_RELEASE_PATH + ';' + BIN_PATH + ';'
        import pyopenpose as op
    else:
        sys.path.append(PYTHON_PATH)
        from openpose import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


class OpenPose:
    def __init__(self):
        self.params = dict()
        self.opWrapper = op.WrapperPython()
        self.set_default_params()
        self.opWrapper.start()

    def set_default_params(self):
        self.params["model_folder"] = MODELS_PATH
        self.params["hand"] = True
        self.params["number_people_max"] = 1
        self.params["keypoint_scale"] = 3
        self.params["render_threshold"] = 0.1
        self.params["disable_blending"] = True
        self.opWrapper.configure(self.params)

    def draw_pose_on_image(self, image):
        return self._process_image(image).cvOutputData

    def get_hand_key_points_from_image(self, image):
        datum = self._process_image(image)

        return datum.handKeypoints[0][0], datum.handKeypoints[1][0]  # Return left and right hand key points

    def _process_image(self, image):
        datum = op.Datum()
        datum.cvInputData = image
        self.opWrapper.emplaceAndPop([datum])

        return datum