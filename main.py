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


HOST = "172.20.10.5"  # Robot IP
PORT = 65438  # The port used by the server


def run_video(online = True):
    # Takes a video path and runs the analysis on each frame
    # Saves the results to the same directory as the video
    videoDebugger = VideoDebugger.VideoDebugger()
    analyser = analyse.Analyse()
    steering_instance = steering.Steering(online, HOST, PORT)
    frame_number = 0

    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920 / 2)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080 / 2)
    print("Video read")

    ball_vector = None
    angle_degrees = None
    signed_angle_degrees = None
    steering_instance.start_belt()
    while True:
        ret, frame = video.read()
        if not ret:
            break
        #print(f"Analysing frame {frame_number}...")

        analyser.analysis_pipeline(frame)
        
        try:
            steering_instance.pick_program(analyser.keypoints, analyser.robot_pos, analyser.robot_vector, analyser.distance_to_closest_border)
        except BallNotFoundError as e:
            print(f"Ball not found: {e}")
        except RobotNotFoundError as e:
            print(f"Robot not found: {e}")
        except Exception as e:
            print(f"Error: {e}")
            
        frame_number += 1
        #print(f"Frame {frame_number} analysed")

        # print(f"Angle between robot and ball: {angle_degrees}")
        # print(f"Signed angle between robot and ball: {signed_angle_degrees}")

        # print("Corners found at: ")
        # print(corners)
        #print(f"{len(analyser.white_ball_keypoints)} balls found")
        #print("Balls found at: ")
        # for keypoint in keypoints:
        #    print(keypoint.pt)
        #    print(keypoint.size)

        # Overlay red vector on robot
        # print(f"Ball vector: {ball_vector}")
        # print(f"Robot vector: {robot_vector}")
        # print(f"Robot pos: {robot_pos}")
        # Display the result

        robot_arrows_on_frame = frame


        height, width = 360, 640
        text_overview = np.zeros((height, width, 3), dtype=np.uint8)



        data_dict = {
            'Robot position': analyser.robot_pos,
            'Robot vector': analyser.robot_vector,
            'Ball vector': ball_vector,
            'Angle': angle_degrees,
            'Signed angle': signed_angle_degrees
        }

        y_offset = 20
        for key, value in data_dict.items():
            text = f"{key}: {value}"
            cv2.putText(text_overview, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

        green_robot_3channel = cv2.cvtColor(
            analyser.green_robot_mask, cv2.COLOR_GRAY2BGR
        )

        # Convert result to 3 channel image
        result_binary = cv2.bitwise_or(analyser.white_mask, analyser.orange_mask)
        result_3channel = cv2.cvtColor(result_binary, cv2.COLOR_GRAY2BGR)

        # Overlay green circle on each keypoint
        for keypoint in analyser.keypoints:
            center = (int(keypoint.pt[0]), int(keypoint.pt[1]))
            radius = int(keypoint.size / 2)
            cv2.circle(result_3channel, center, radius, (0, 255, 0), 4)

        # Overlay red circle on red robot
        # if analyser.red_pos is not None:
        #    center = (int(analyser.red_pos[0]), int(analyser.red_pos[1]))
        #    radius = 30
        #    cv2.circle(red_robot_3channel, center, radius, (0, 0, 255), 4)
        #    print(f"Red robot at {center}")
        # Overlay green circle on green robot
        if analyser.robot_pos is not None:
            center = (int(analyser.robot_pos[0]), int(analyser.robot_pos[1]))
            radius = 30
            print(f"Green robot at {center}")
            cv2.circle(green_robot_3channel, center, radius, (0, 255, 0), 4)

        if analyser.robot_vector is not None and analyser.robot_pos is not None:
            robot_vector_end = analyser.robot_pos + analyser.robot_vector
            # Cast to int for drawing
            robot_pos = analyser.robot_pos.astype(int)
            robot_vector_end = robot_vector_end.astype(int)
            cv2.arrowedLine(
                robot_arrows_on_frame,
                tuple(robot_pos),
                tuple(robot_vector_end),
                (0, 0, 255),
                2,
            )

        if steering_instance.ball_vector is not None and analyser.robot_pos is not None:
            ball_vector_end = analyser.robot_pos + steering_instance.ball_vector
            # Cast to int for drawing
            robot_pos = analyser.robot_pos.astype(int)
            ball_vector_end = ball_vector_end.astype(int)
            cv2.arrowedLine(
                robot_arrows_on_frame,
                tuple(robot_pos),
                tuple(ball_vector_end),
                (255, 0, 0),
                2,
            )

        # Write text in the bottom left corner of the image
        if angle_degrees is not None and signed_angle_degrees is not None:
            text = f"Angle: {angle_degrees}  Signed Angle: {signed_angle_degrees}"
            cv2.putText(
                robot_arrows_on_frame,
                text,
                (10, robot_arrows_on_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        videoDebugger.write_video("result", result_3channel, True)
        im1 = cv2.resize(robot_arrows_on_frame, (640, 360))

        im2 = cv2.resize(result_3channel, (640, 360))

        im3 = cv2.resize(text_overview, (640, 360))

        im4 = cv2.resize(green_robot_3channel, (640, 360))

        # Split the frame into four equal parts
        hstack1 = np.hstack((im1, im2))
        hstack2 = np.hstack((im3, im4))
        combined_images = np.vstack((hstack1, hstack2))
        cv2.imshow("Combined Images", combined_images)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            videoDebugger.close_videos()
            video.release()
            cv2.destroyAllWindows()
            steering_instance.stop_belt()

            break

    video.release()
    cv2.destroyAllWindows()


# Run video analysis
if "offline" in sys.argv:
    run_video(False)
else:
    run_video()

