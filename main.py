#!/usr/bin/env python3
import analyse
import sys
import cv2
import numpy as np
import math
import time
from analyse import BallNotFoundError, RobotNotFoundError
import VideoDebugger
import analyse
import steering
import videoOutput


HOST = "172.20.10.5"  # Robot IP
PORT = 65438  # The port used by the server



def run_video(online = True):
    # Takes a video path and runs the analysis on each frame
    # Saves the results to the same directory as the video
    video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    videoDebugger = VideoDebugger.VideoDebugger()
    analyser = analyse.Analyse()
    steering_instance = steering.Steering(online, HOST, PORT)
    
    data_dict = {
            'Robot position': analyser.robot_pos,
            'Robot vector': analyser.robot_vector,
            'Ball vector': steering_instance.ball_vector,
            'Angle': steering_instance.angle_degrees,
            'Signed angle': steering_instance.signed_angle_degrees,
            'Close to Ball': steering_instance.close_to_ball,
            'Time to switch target': steering_instance.time_to_switch_target,
        }
    
    video_output = videoOutput.VideoOutput(analyser, steering_instance, videoDebugger, data_dict)
    frame_number = 0

    video = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920 / 2)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080 / 2)
    print("Video read")

    steering_instance.start_belt()
    while True:
        ret, frame = video.read()
        if not ret:
            break

        analyser.analysis_pipeline(frame)
        
        try:
            steering_instance.pick_program(analyser.keypoints, analyser.robot_pos, analyser.robot_vector, analyser.distance_to_closest_border)
        except BallNotFoundError as e:
            print(f"Ball not found: {e}")
        except RobotNotFoundError as e:
            print(f"Robot not found: {e}")
        except Exception as e:
            print(f"Error: {e}")
            
        video_output.showFrame(frame)
        
        frame_number += 1
    
        if cv2.waitKey(25) & 0xFF == ord("q"):
            videoDebugger.close_videos()
            video.release()
            cv2.destroyAllWindows()
            steering_instance.stop_belt()
            steering_instance.disconnect()

            break

    video.release()
    cv2.destroyAllWindows()


# Run video analysis
if "offline" in sys.argv:
    run_video(False)
else:
    run_video()

