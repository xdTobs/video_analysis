import cv2
import os
import datetime
from typing import List, Dict
import numpy as np
from analyse import Analyse
from steering import Steering
from VideoDebugger import VideoDebugger


class VideoOutput():
    def __init__(self, analyser, steering_instance, videoDebugger, data_dict: Dict[str, any]):
        self.analyser = analyser
        self.steering_instance = steering_instance
        self.videoDebugger = videoDebugger
        self.data_dict = data_dict

    def update_data_dict(self, new_data: Dict[str, any]):
        self.data_dict.update(new_data) 

    def showFrame(self,frame):
        robot_arrows_on_frame = frame


        height, width = 360, 640
        text_overview = np.zeros((height, width, 3), dtype=np.uint8)
        
        y_offset = 20
        for key, value in self.data_dict.items():
            self.update_data_dict({key: value})
            text = f"{key}: {value}"
            cv2.putText(text_overview, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

        green_robot_3channel = cv2.cvtColor(
            self.analyser.green_robot_mask, cv2.COLOR_GRAY2BGR
        )

        # Convert result to 3 channel image
        result_binary = cv2.bitwise_or(self.analyser.white_mask, self.analyser.orange_mask)
        result_3channel = cv2.cvtColor(result_binary, cv2.COLOR_GRAY2BGR)

        # Overlay green circle on each keypoint
        for keypoint in self.analyser.keypoints:
            center = (int(keypoint.pt[0]), int(keypoint.pt[1]))
            radius = int(keypoint.size / 2)
            cv2.circle(result_3channel, center, radius, (0, 255, 0), 4)

        if self.analyser.robot_pos is not None:
            center = (int(self.analyser.robot_pos[0]), int(self.analyser.robot_pos[1]))
            radius = 30
            print(f"Green robot at {center}")
            cv2.circle(green_robot_3channel, center, radius, (0, 255, 0), 4)

        if self.analyser.robot_vector is not None and self.analyser.robot_pos is not None:
            robot_vector_end = self.analyser.robot_pos + self.analyser.robot_vector
            # Cast to int for drawing
            robot_pos = self.analyser.robot_pos.astype(int)
            robot_vector_end = robot_vector_end.astype(int)
            cv2.arrowedLine(
                robot_arrows_on_frame,
                tuple(robot_pos),
                tuple(robot_vector_end),
                (0, 0, 255),
                2,
            )

        if self.steering_instance.ball_vector is not None and self.analyser.robot_pos is not None:
            ball_vector_end = self.analyser.robot_pos + self.steering_instance.ball_vector
            # Cast to int for drawing
            robot_pos = self.analyser.robot_pos.astype(int)
            ball_vector_end = ball_vector_end.astype(int)
            cv2.arrowedLine(
                robot_arrows_on_frame,
                tuple(robot_pos),
                tuple(ball_vector_end),
                (255, 0, 0),
                2,
            )

       

        self.videoDebugger.write_video("result", result_3channel, True)
        im1 = cv2.resize(robot_arrows_on_frame, (640, 360))

        im2 = cv2.resize(result_3channel, (640, 360))

        im3 = cv2.resize(text_overview, (640, 360))

        im4 = cv2.resize(green_robot_3channel, (640, 360))

        # Split the frame into four equal parts
        hstack1 = np.hstack((im1, im2))
        hstack2 = np.hstack((im3, im4))
        combined_images = np.vstack((hstack1, hstack2))
        cv2.imshow("Combined Images", combined_images)