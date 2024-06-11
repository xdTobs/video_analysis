#!/usr/bin/env python3
import analyse
import sys
import cv2
import numpy as np
import math
import time
import RobotInterface

import VideoDebugger
import analyse
import steering


HOST = "172.20.10.3"  # Robot IP
PORT = 65438  # The port used by the server


def run_video(robotInterface: RobotInterface.RobotInterface):
    # Takes a video path and runs the analysis on each frame
    # Saves the results to the same directory as the video
    videoDebugger = VideoDebugger.VideoDebugger()
    analyser = analyse.Analyse()
    frame_number = 0

    video = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920 / 2)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080 / 2)
    print("Video read")

    try:
        robotInterface.send_command("belt", 0, speedPercentage=100)
    except ConnectionError as e:
        print("Robot not connected", e)
    except Exception as e:
        print("Error sending: ", e)

    ball_vector = None
    angle_degrees = None
    signed_angle_degrees = None

    while True:
        ret, frame = video.read()
        if not ret:
            break
        print(f"Analysing frame {frame_number}...")

        analyser.analysis_pipeline(frame)

        frame_number += 1
        print(f"Frame {frame_number} analysed")

        steering_instance = steering.Steering()
        if (
            analyser.robot_pos is not None
            and analyser.robot_vector is not None
            and analyser.white_ball_keypoints is not None
        ):
            try:
                ball_vector = steering_instance.find_ball_vector(
                    analyser.white_ball_keypoints,
                    analyser.robot_pos,
                    analyser.robot_vector,
                )

                dist_to_ball = np.linalg.norm(ball_vector)
                print(f"Ball vector: {ball_vector}")
                print(f"Ball vector length: {dist_to_ball}")
                if dist_to_ball < 100:
                    print("Ball is close")
                else:
                    print("Ball is not close")

                steering_instance.catch_ball(ball_vector, analyser.robot_vector)
            except analyse.BallNotFoundError as e:
                print("No balls found", e)
            except analyse.RobotNotFoundError as e:
                print("No robot found", e)

        def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
            if v1 is None or v2 is None:
                return 0
            dot_prod = np.dot(v1, v2)
            magnitude_v1 = np.linalg.norm(v1)
            magnitude_v2 = np.linalg.norm(v2)
            cos_theta = dot_prod / (magnitude_v1 * magnitude_v2)
            return np.arccos(cos_theta)

        def angle_between_vectors_signed(v1: np.ndarray, v2: np.ndarray) -> float:
            if v1 is None or v2 is None:
                return 0
            # Dot product of two vectors
            dot_prod = np.dot(v1, v2)
            # Determinant (pseudo cross-product) in 2D
            det = v1[0] * v2[1] - v1[1] * v2[0]
            # Angle in radians
            angle_radians = np.arctan2(det, dot_prod)
            return angle_radians

        if ball_vector is not None:
            signed_angle_radians = angle_between_vectors_signed(analyser.robot_vector, ball_vector)  # type: ignore
            signed_angle_degrees = math.degrees(signed_angle_radians)
            angle_radians = angle_between_vectors(analyser.robot_vector, ball_vector)  # type: ignore
            angle_degrees = math.degrees(angle_radians)

        try:
            if (
                angle_degrees is not None
                and signed_angle_degrees is not None
                and angle_degrees < 2
                and analyser.robot_pos is not None
                and ball_vector is not None
            ):
                robotInterface.send_command("move", 30, 30)
            else:
                print(f"Turning {signed_angle_degrees} degrees")
                if signed_angle_degrees is not None:
                    robotInterface.send_command(
                        "turn", signed_angle_degrees * -1 / 3, 15
                    )
        except ConnectionError as e:
            print("Robot not connected", e)
        except Exception as e:
            print("Error sending: ", e)

        # print(f"Angle between robot and ball: {angle_degrees}")
        # print(f"Signed angle between robot and ball: {signed_angle_degrees}")

        # print("Corners found at: ")
        # print(corners)
        print(f"{len(analyser.white_ball_keypoints)} balls found")
        print("Balls found at: ")
        # for keypoint in keypoints:
        #    print(keypoint.pt)
        #    print(keypoint.size)

        # Overlay red vector on robot
        # print(f"Ball vector: {ball_vector}")
        # print(f"Robot vector: {robot_vector}")
        # print(f"Robot pos: {robot_pos}")
        # Display the result

        robot_arrows_on_frame = frame

        red_robot_3channel = cv2.cvtColor(analyser.red_robot_mask, cv2.COLOR_GRAY2BGR)
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

        if ball_vector is not None and analyser.robot_pos is not None:
            ball_vector_end = analyser.robot_pos + ball_vector
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

        im3 = cv2.resize(red_robot_3channel, (640, 360))

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
            robotInterface.send_command("belt", 0, speedPercentage=-100)
            time.sleep(5)
            robotInterface.send_command("belt", 0, speedPercentage=0)

            break

    video.release()
    cv2.destroyAllWindows()


# Run video analysis
if "offline" in sys.argv:
    robotInterface = RobotInterface.RobotInterface(HOST, PORT, online=False)
else:
    robotInterface = RobotInterface.RobotInterface(HOST, PORT, online=True)

try:
    robotInterface.connect()
    # robotInterface.send_command("turn",-90,50)
except ConnectionError as e:
    print("Robot not connected", e)
run_video(robotInterface=robotInterface)
