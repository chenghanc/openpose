import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
# Append to syspath the path to openpose python build
#sys.path.append('../../python')
sys.path.append('/home/nechk/NECHK-Results/helmet2/emotion/openpose/build/python')

try:
    from openpose import pyopenpose as op
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["logging_level"] = 4
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_folder"] = "../../../models/"
params["model_pose"] = "BODY_25B"   # BODY_25
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
params["render_pose"] = 2
params["disable_blending"] = False
params["heatmaps_add_parts"] = False
params["heatmaps_add_PAFs"] = False
params["display"] = 0
params["write_images"] = "output_images/"
params["write_images_format"] = "jpg"
params["write_json"] = "output_jsons/"
#params["face"] = True
#params["hand"] = True

# Starting OpenPose. Construct OpenPose object allocates GPU memory
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Process Image
datum = op.Datum()
imageToProcess = cv2.imread('../../../examples/media/COCO_val2014_000000000241.jpg')
datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop(op.VectorDatum([datum]))

# Display Image
print("Body keypoints: \n" + str(datum.poseKeypoints))
print("Face keypoints: \n" + str(datum.faceKeypoints))
print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
cv2.waitKey(0)
