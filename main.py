#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import analyse
import sys
import cv2
from analyse import BallNotFoundError, RobotNotFoundError
import VideoDebugger
import analyse
import steering
import videoOutput
import platform


def run_video(host, webcam_index, online, port=65438):
    # Takes a video path and runs the analysis on each frame
    # darwin is mac
    if platform.system() == "Windows":
        video = cv2.VideoCapture(webcam_index)
    elif platform.system() == "Linux" or platform.system() == "Darwin":
        video = cv2.VideoCapture(webcam_index)
    else:
        raise Exception("Unsupported platform. Please use Windows, Linux or Mac.")
    videoDebugger = VideoDebugger.VideoDebugger()
    analyser = analyse.Analyse()
    steering_instance = steering.Steering(online, host, port)

    data_dict = {
        "Robot position": analyser.robot_pos,
        "Robot vector": analyser.robot_vector,
        "Ball vector": steering_instance.ball_vector,
        "Angle": steering_instance.angle_degrees,
        "Signed angle": steering_instance.signed_angle_degrees,
        "Close to Ball": steering_instance.close_to_ball,
        "Time to switch target": steering_instance.time_to_switch_target,
        "Distance to closest border": analyser.distance_to_closest_border,
    }

    video_output = videoOutput.VideoOutput(
        analyser, steering_instance, videoDebugger, data_dict
    )
    frame_number = 0

    # video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
    print("Video read")

    steering_instance.start_belt()
    while True:
        ret, frame = video.read()
        if not ret:
            break

        analyser.analysis_pipeline(frame)

        data_dict["Distance to closest border"] = analyser.distance_to_closest_border
        
        try:
            steering_instance.pick_program(
                analyser.keypoints,
                analyser.robot_pos,
                analyser.robot_vector,
                analyser.distance_to_closest_border,
            )
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
            if online:
                steering_instance.stop_belt()
                steering_instance.disconnect()
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    load_dotenv(override=True)
    HOST = os.getenv("HOST")
    PORT = os.getenv("PORT")
    WEBCAM_INDEX = os.getenv("WEBCAM_INDEX")
    IS_GOAL_RIGHT_SIDE = os.getenv("IS_GOAL_RIGHT_SIDE")
    is_offline = "offline" in sys.argv

    print("HOST: ", HOST)
    print("PORT: ", PORT)
    print("WEBCAM_INDEX: ", WEBCAM_INDEX)
    print("IS_GOAL_RIGHT_SIDE: ", IS_GOAL_RIGHT_SIDE)
    should_quit = False
    if HOST is None:
        print("No HOST provided in .env file")
        should_quit = True
    if PORT is None:
        print("No PORT provided in .env file. 65438 is the most common port")
        should_quit = True
    if WEBCAM_INDEX is None:
        print("No WEBCAM_INDEX provided in .env file. 0 or 1 is the most common index")
        should_quit = True
    if IS_GOAL_RIGHT_SIDE is None:
        print("No IS_GOAL_RIGHT_SIDE provided in .env file. True or False")
        should_quit = True

    print("HOST: ", HOST)
    print("PORT: ", PORT)
    print("WEBCAM_INDEX: ", WEBCAM_INDEX)
    print("is_offline: ", is_offline)
    if should_quit:
        print("Exiting... Please provide the missing values in the .env file")
        sys.exit(1)

    run_video(
        host=HOST, webcam_index=str(WEBCAM_INDEX), online=not is_offline, port=int(PORT)
    )
