import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import time

def display(datums):
    datum = datums[0]
    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    key = cv2.waitKey(15)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
    return (key == 27)

def printKeypoints(datums):
    datum = datums[0]
    print("Body keypoints: \n" + str(datum.poseKeypoints))

dir_path = os.path.dirname(os.path.realpath(__file__))
# Append to syspath the path to openpose python build
sys.path.append('../../python')

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

# Read frames on directory
input_source = "../../../examples/media/"

imagePaths = op.get_images_on_directory(input_source);
start = time.time()

count = 0
# Process and display images
for imagePath in imagePaths:
    datum = op.Datum()
    imageToProcess = cv2.imread(imagePath)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # Display Image
    # Enable to disable the visual display
    no_display = False
    printKeypoints(op.VectorDatum([datum]))
    if not no_display:
        display(op.VectorDatum([datum]))

    cv2.imwrite("output_images/frame_%d.jpg" % count, datum.cvOutputData)
    count += 1

end = time.time()
print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")

# Keypoint Ordering in Python
poseModel = op.PoseModel.BODY_25B
print("Get Body Part Mapping:   ", op.getPoseBodyPartMapping(poseModel), "\n")
print("Get Number Body Parts:   ", op.getPoseNumberBodyParts(poseModel), "\n")
print("Get Part Pairs:          ", op.getPosePartPairs(poseModel), "\n")
print("Get Map Index:           ", op.getPoseMapIndex(poseModel), "\n")
