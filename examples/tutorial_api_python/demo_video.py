import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import json

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
params["model_pose"] = "BODY_25B"   # BODY_25
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
            print("Body keypoints: \n" + str(datum.poseKeypoints))
            cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break
        else:
            break

    cap.release()
    video.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(e)
    sys.exit(-1)

print("COMPLETED!!!")
