import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import json
import time

def display(datums,cap):
    datum = datums[0]
    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        cap.release()
        #video.release()
        cv2.destroyAllWindows()
    return (key == 27)

def printKeypoints(datums):
    datum = datums[0]
    print("Body keypoints: \n" + str(datum.poseKeypoints))
    print("Face keypoints: \n" + str(datum.faceKeypoints))
    print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
    print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

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
params["disable_blending"] = True
params["heatmaps_add_parts"] = False
params["heatmaps_add_PAFs"] = False
params["display"] = 0
#params["write_video"] = "output/result.avi"
#params["write_images"] = "output_images/"
params["write_json"] = "output_jsons/"
#params["face"] = True
#params["hand"] = True
#params["number_people_max"] = 1

'''
# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item
'''

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

try:
    # Starting OpenPose. Construct OpenPose object allocates GPU memory
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    # Process Video
    input_source = "../../../examples/media/video.avi"
    datum = op.Datum()
    cap = cv2.VideoCapture(input_source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video=None
    count=0
    start = time.time()
    while (cap.isOpened()):
        count=count+1
        hasframe, frame = cap.read()

        if hasframe== True:
            datum.cvInputData = frame
            datum.name=str(count)
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            opframe=datum.cvOutputData
            height, width, layers = opframe.shape
            if video == None:
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                video = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))
            video.write(opframe)

            # Display Image
            # Enable to disable the visual display
            no_display = False
            #print("Body keypoints: \n" + str(datum.poseKeypoints))
            #cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
            #key = cv2.waitKey(1)
            #if key & 0xFF == ord('q') or key == 27:
            #    cap.release()
            #    video.release()
            #    cv2.destroyAllWindows()
            #    break
            printKeypoints(op.VectorDatum([datum]))
            if not no_display:
                display(op.VectorDatum([datum]),cap)

            # Start of Applications
            instances=len(datum.poseKeypoints)

            print("Number of instances per image:    ", instances)
            for i in range(0,instances):
                print("Instance : ", i)
                RbaseY = datum.poseKeypoints[i][4][1]  # 4: 'RWrist' 3: 'RElbow' (0,1,2) = (x,y,c)
                LbaseY = datum.poseKeypoints[i][7][1]  # 7: 'LWrist' 6: 'LElbow' (0,1,2) = (x,y,c)
                NoseY  = datum.poseKeypoints[i][0][1]  # 0: 'Nose'               (0,1,2) = (x,y,c)
                print("NoseY = {:.1f}, RbaseY = {:.1f}, LbaseY = {:.1f}".format(NoseY, RbaseY, LbaseY))
                if (NoseY==0.0 or RbaseY==0.0 or LbaseY==0.0):
                    print("State : Not in ROI")
                elif (RbaseY < NoseY or LbaseY < NoseY):
                    print("State : Hands Up")
                else:
                    print("State : Normal" + "\n")
            # End of Applications

        else:
            break

    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")

    # Keypoint Ordering in Python
    poseModel = op.PoseModel.BODY_25
    print("Get Body Part Mapping:   ", op.getPoseBodyPartMapping(poseModel), "\n")
    print("Get Number Body Parts:   ", op.getPoseNumberBodyParts(poseModel), "\n")
    print("Get Part Pairs:          ", op.getPosePartPairs(poseModel), "\n")
    print("Get Map Index:           ", op.getPoseMapIndex(poseModel), "\n")

except Exception as e:
    print(e)
    sys.exit(-1)

print("COMPLETED!!!")
