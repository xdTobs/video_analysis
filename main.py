#!/usr/bin/env python3
import collections
import os
from time import sleep
import time
from dotenv import load_dotenv
import sys
import cv2
import numpy as np
from analyse import BorderNotFoundError, RobotNotFoundError, find_corners, isol_borders
import VideoDebugger
import analyse
import steering
import videoOutput
import webcam


def run_video(host, webcam_index, online, port=65438):
    # Takes a video path and runs the analysis on each frame
    video = webcam.open_webcam_video(webcam_index)
    videoDebugger = VideoDebugger.VideoDebugger()

    # find most common corners
    corners = None
    corners_list = []
    i = 0
    img = None
    while True:
        start_time = time.time()
        ret, frame = video.read()
        if not ret:
            print("could not read frame when finding corners. Exiting...")
            sys.exit(1)
        mask = isol_borders(frame, "border")
        try:
            corners = find_corners(mask)
            for corner in corners:
                cv2.circle(frame, corner, 5, (0, 0, 255), -1)
            i += 1
        except BorderNotFoundError as e:
            print("could not find border, skipping frame", e)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        print(corners)
        corners_list.append(corners)

        # stack = np.vstack((frame, mask))

        # cv2.imshow("frame", stack)
        if cv2.waitKey(25) & 0xFF == ord("q") or i > 10:
            img = frame
            # cv2.destroyAllWindows()
            break
        end_time = time.time()
        print(f"Frame time borders: {end_time - start_time} seconds")

    corners_tuple_list = []
    for arr in corners_list:
        corners_tuple_list.append(tuple(map(tuple, arr)))
    corners = collections.Counter(corners_tuple_list).most_common()[0][0]
    # draw corners on image

    for corner in corners:
        cv2.circle(img, corner, 5, (0, 0, 255), -1)

    cv2.imwrite("corners.jpg", img)
    # Convert the most common group back to a numpy array
    # corners = np.array(corners_counter[0], dtype=np.int32)

    analyser = analyse.Analyse()
    steering_instance = steering.Steering(online, host, port)

    data_dict = {
        "Robot position": analyser.robot_pos,
        "Robot vector": analyser.robot_vector,
        "Ball vector": steering_instance.steering_vector,
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
    print("Video read")

    steering_instance.start_belt()
    while True:
        start_time = time.time()
        ret, frame = video.read()
        if not ret:
            break

        analyser.analysis_pipeline(frame)

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
                border_mask=analyser.border_mask,
            )
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
        end_time = time.time()
        print(f"Frame time main analysis: {end_time - start_time} seconds")

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

    # pr = cProfile.Profile()
    # pr.enable()  # Start profiling
    run_video(
        host=HOST, webcam_index=WEBCAM_INDEX, online=True, port=int(PORT)
    )

    # pr.disable()  # Stop profiling

    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    # ps.print_stats()  # Print the profiling results
    # with open("profile_results.txt", "w") as f:
    #     f.write(s.getvalue())
