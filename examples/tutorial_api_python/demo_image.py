import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import math

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
params["model_pose"] = "BODY_25"   # BODY_25B
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
imageToProcess = cv2.imread('../../../examples/media/falling1.jpg')
datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop(op.VectorDatum([datum]))

# Display Image
print("Body keypoints: \n" + str(datum.poseKeypoints))
print("Body keypoints: \n" + str(datum.poseKeypoints[0][2][2]))
#print("Face keypoints: \n" + str(datum.faceKeypoints))
#print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
#print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))

# Start of Applications
instances=len(datum.poseKeypoints)

print("Number of instances per image:    ", instances)
for i in range(0,instances):
    print("Instance : ", i)
    RbaseY = datum.poseKeypoints[i][4][1]  # 4: 'RWrist' 3: 'RElbow' (0,1,2) = (x,y,c)
    LbaseY = datum.poseKeypoints[i][7][1]  # 7: 'LWrist' 6: 'LElbow' (0,1,2) = (x,y,c)
    NoseY  = datum.poseKeypoints[i][0][1]  # 0: 'Nose'               (0,1,2) = (x,y,c)
    NoseX  = datum.poseKeypoints[i][0][0]  # 0: 'Nose'               (0,1,2) = (x,y,c)
    RAnkleY =datum.poseKeypoints[i][11][1] # 11:'RAnkle'             (0,1,2) = (x,y,c)
    RAnkleX =datum.poseKeypoints[i][11][0] # 11:'RAnkle'             (0,1,2) = (x,y,c)
    LAnkleY =datum.poseKeypoints[i][14][1] # 14:'LAnkle'             (0,1,2) = (x,y,c)
    LAnkleX =datum.poseKeypoints[i][14][0] # 14:'LAnkle'             (0,1,2) = (x,y,c)
    s11= (RAnkleX, RAnkleY)
    s14= (LAnkleX, LAnkleY)
    s0 = (NoseX, NoseY)
    barX = (RAnkleX + LAnkleX) * .5
    barY = (RAnkleY + LAnkleY) * .5
    # Fall
    # The angle between the centerline of the body and the ground
    theta= np.arctan(abs((NoseY-barY)/(NoseX-barX))) # angle in degree = angle in radian * 180/pi
    theta=theta*(180/np.pi)
    theta0=45
    #print(math.degrees(math.pi/2))
    #print(math.degrees(theta))
    print(s11, s14)
    print("NoseX = {:.1f}, NoseY = {:.1f}, barX = {:.1f}, barY = {:.1f}, A = {:.3f}".format(NoseX, NoseY, barX, barY, theta))
    if (NoseX==0.0 and NoseY==0.0):
        print("State : Not in ROI")
    elif (theta < theta0):
        print("State : Fall")
    else:
        print("State : Normal")
    '''
    # Hands Up
    print("NoseY = {:.1f}, RbaseY = {:.1f}, LbaseY = {:.1f}".format(NoseY, RbaseY, LbaseY))
    if (NoseY==0.0 or RbaseY==0.0 or LbaseY==0.0):
        print("State : Not in ROI")
    elif (RbaseY < NoseY or LbaseY < NoseY):
        print("State : Hands Up")
    else:
        print("State : Normal" + "\n")
    '''
# End of Applications

cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
cv2.waitKey(0)

# Keypoint Ordering in Python
poseModel = op.PoseModel.BODY_25
print("Get Body Part Mapping:   ", op.getPoseBodyPartMapping(poseModel), "\n")
print("Get Number Body Parts:   ", op.getPoseNumberBodyParts(poseModel), "\n")
print("Get Part Pairs:          ", op.getPosePartPairs(poseModel), "\n")
print("Get Map Index:           ", op.getPoseMapIndex(poseModel), "\n")
