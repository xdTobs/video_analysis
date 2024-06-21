#!/usr/bin/env python3
import cProfile
import io
import os
import pstats
import time
from dotenv import load_dotenv
import sys
import cv2
import numpy as np
from analyse import RobotNotFoundError
import VideoDebugger
import analyse
import steering
import videoOutput
import webcam
from utils import angle_between_vectors_signed, coordinates_to_vector


def run_video(host, webcam_index, online, port=65438):
    # Takes a video path and runs the analysis on each frame
    video = webcam.open_webcam_video(webcam_index)
    videoDebugger = VideoDebugger.VideoDebugger()

    analyser = analyse.Analyse()
    steering_instance = steering.Steering(online, host, port)

    data_dict = {
        "Robot position": analyser.robot_pos,
        "Robot vector": analyser.robot_vector,
        # "Ball vector": steering_instance.ball_vector,
        "Is ball close to border": steering_instance.is_ball_close_to_border,
        "Angle": steering_instance.angle_degrees,
        "Signed angle": steering_instance.signed_angle_degrees,
        "Robot close to Ball": steering_instance.close_to_ball,
        "Time to switch target": steering_instance.time_to_switch_target,
        "Robot distance to closest border": analyser.distance_to_closest_border,
        "Collecting balls": steering_instance.is_collecting_balls,
    }

    video_output = videoOutput.VideoOutput(
        analyser, steering_instance, videoDebugger, data_dict
    )
    frame_number = 0
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)

    # https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv
    video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("Video read")
    steering_instance.start_belt()

    # find most common corners
    found_corners = None
    corners_list = []
    has_found_corners = False
    while True:
        start_time = time.time()
        init_time = start_time
        ret, frame = video.read()

        if not ret:
            break

        prev_time = time.time()
        print(f"FTAN1: {prev_time - start_time} seconds")
        start_time = time.time()

        analyser.analysis_pipeline(image=frame, has_found_corners=has_found_corners)

        if analyser.robot_pos is not None and steering_instance.target_ball is not None:
            ball_vector = coordinates_to_vector(
                analyser.robot_pos, steering_instance.target_ball
            )
            ball_distance = np.linalg.norm(ball_vector)

            steering_instance.set_speed(ball_distance, angle_between_vectors_signed(analyser.robot_vector,ball_vector) )
        

        prev_time = time.time()
        print(f"FTAN2: {prev_time - start_time} seconds")
        start_time = time.time()
        print(found_corners)
        if not has_found_corners:
            corners_list.append(analyser.corners)
            if len(corners_list) == 10:
                corners_list = np.array(corners_list)
                corners = np.median(corners_list, axis=0)
                corners = corners.astype(int)
                has_found_corners = True
            else:
                continue

        prev_time = time.time()
        print(f"FTAN3: {prev_time - start_time} seconds")
        start_time = time.time()

        try:
            steering_instance.pick_program_pipeline(
                analyser.keypoints,
                analyser.robot_pos,
                analyser.robot_vector,
                analyser.distance_to_closest_border,
                analyser.border_vector,
                analyser.corners,
                analyser.dropoff_coords,
                analyser.safepoint_list,
                analyser.small_goal_coords,
                border_mask=analyser.border_mask,
            )
        except RobotNotFoundError as e:
            print(f"Robot not found: {e}")
        except Exception as e:
            print(f"Error: {e}")
        prev_time = time.time()
        print(f"FTAN4: {prev_time - start_time} seconds")
        start_time = time.time()

        video_output.showFrame(frame)

        frame_number += 1

        key = cv2.waitKey(25) & 0xFF
        if key == ord("q"):
            videoDebugger.close_videos()
            video.release()
            cv2.destroyAllWindows()
            if online:
                steering_instance.ejaculate()
                steering_instance.disconnect()
            break
        elif key == ord("p"):
            cv2.waitKey(0)
        prev_time = time.time()
        print(f"FTAN5: {prev_time - start_time} seconds")
        start_time = time.time()
        print(f"FTANX: {prev_time - init_time} seconds")
        sys.stdout.flush()

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

    pr = cProfile.Profile()
    pr.enable()  # Start profiling
    run_video(
        host=HOST, webcam_index=WEBCAM_INDEX, online=not is_offline, port=int(PORT)
    )

    pr.disable()  # Stop profiling

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()  # Print the profiling results
    with open("profile_results.txt", "w") as f:
        f.write(s.getvalue())
