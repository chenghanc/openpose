import sys
import os
from sys import platform
import numpy as np
from imutils.video import FPS
import cv2
import time

class Main:

    def __init__(self, source):
        self.stream = cv2.VideoCapture(source)
        self.op_params = self.set_op_params()
    
    def set_op_params(self):
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
        #params["write_video"] = "output/result.avi"
        #params["write_images"] = "output_images/"
        params["write_json"] = "output_jsons/"
        params["write_video_fps"] = -1
        params["part_candidates"] = True
        #params["face"] = True
        #params["hand"] = True
        #params["number_people_max"] = 1
        return params

    def start(self):
        
        #fps = FPS().start()
        start = time.time()
        video=None
        count=0

        try:
            # Starting OpenPose. Construct OpenPose object allocates GPU memory
            opWrapper = op.WrapperPython()
            opWrapper.configure(self.op_params)
            opWrapper.start()
            
            datum = op.Datum()

            fps = self.stream.get(cv2.CAP_PROP_FPS)
            framecount = self.stream.get(cv2.CAP_PROP_FRAME_COUNT)
            print('Total frames in this video: ' + str(framecount))
            
            if (self.stream.isOpened() == False):
                sys.exit("Error opening video stream or file")
            
            while self.stream.isOpened():

                count=count+1
                hasframe, frame = self.stream.read()
                
                if frame is None or not hasframe:
                    print("Error while opening capture device...")
                    break
                
                if hasframe == True:
                    
                    datum.cvInputData = frame
                    datum.name=str(count)
                    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                    
                    opframe=datum.cvOutputData
                    height, width, layers = opframe.shape
                    
                    if video == None:
                        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                        video = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

                    no_display = False

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
                        '''
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
                            cv2.putText(opframe, 'Hands Up', (30, 180),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            print("State : Normal" + "\n")

                    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", opframe)
                    video.write(opframe)
                    # End of Applications

                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q') or key == 27:
                        break

                else:
                    break

            self.stream.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(e)
            sys.exit(-1)

        print("COMPLETED!!!")

if __name__ == "__main__":
    
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
    
    # read video
    if len(sys.argv) > 1:
        source = str(sys.argv[1])
    else:
        source = r'/path/to/Part4/test1.mp4'

    main = Main(source=source)
    main.start()
