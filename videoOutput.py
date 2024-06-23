import cv2
from typing import Dict
import numpy as np
from analyse import Analyse
from VideoDebugger import VideoDebugger
from steering import Steering


class VideoOutput:
    def __init__(
        self, analyser, steering_instance, videoDebugger, data_dict: Dict[str, any]
    ):
        self.analyser: Analyse = analyser
        self.steering_instance: Steering = steering_instance
        self.videoDebugger: VideoDebugger = videoDebugger
        self.data_dict = data_dict

    def update_data_dict(self):
        self.data_dict["Angle"] = self.steering_instance.angle_degrees
        self.data_dict["Signed angle"] = self.steering_instance.signed_angle_degrees
        self.data_dict["Close to Ball"] = self.steering_instance.close_to_ball
        self.data_dict["Time to switch target"] = (
            self.steering_instance.time_to_switch_target
        )
        self.data_dict["Robot distance to closest border"] = (
            self.analyser.distance_to_closest_border
        )
        self.data_dict["Is ball close to border"] = (
            self.steering_instance.is_ball_close_to_border
        )
        self.data_dict["Collecting balls"] = self.steering_instance.is_collecting_balls
        self.data_dict["State"] = self.steering_instance.state.__class__.__name__
        self.data_dict["Is ball close to middle cross"] = self.analyser.is_ball_close_to_middle

    def showFrame(self, frame):
        self.update_data_dict()

        robot_arrows_on_frame = frame

        height, width = 2 * 360, 2 * 640
        text_overview = cv2.cvtColor(
            self.analyser.white_average.astype(np.uint8), cv2.COLOR_GRAY2BGR
        )
        border_mask = cv2.cvtColor(self.analyser.border_mask, cv2.COLOR_GRAY2BGR)

        y_offset = 20
        for key, value in self.data_dict.items():
            text = f"{key}: {value}"
            cv2.putText(
                border_mask,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 127, 255),
                1,
            )
            y_offset += 20

        green_robot_3channel = cv2.cvtColor(
            self.analyser.green_robot_mask, cv2.COLOR_GRAY2BGR
        )
        cross_rect, _ = Analyse.find_cross_bounding_rectangle(self.analyser.border_mask)

        result_binary = cv2.bitwise_or(
            self.analyser.white_mask, self.analyser.orange_mask
        )
        # result_binary = self.analyser.orange_mask
        result_3channel = cv2.cvtColor(result_binary, cv2.COLOR_GRAY2BGR)
        # result_3channel = cv2.cvtColor(
        # self.analyser.white_average.astype(np.uint8), cv2.COLOR_GRAY2BGR
        # )
        for keypoint in self.analyser.keypoints:
            center = (int(keypoint.pt[0]), int(keypoint.pt[1]))
            radius = int(keypoint.size / 2)
            cv2.circle(result_3channel, center, radius, (0, 255, 0), 4)

        if self.analyser.robot_pos is not None:
            center = (int(self.analyser.robot_pos[0]), int(self.analyser.robot_pos[1]))
            radius = 30
            cv2.circle(green_robot_3channel, center, radius, (255, 0, 0), 4)

        if self.analyser.green_points_not_translated is not None:
            for point in self.analyser.green_points_not_translated:
                cv2.circle(
                    green_robot_3channel,
                    (int(point[0]), int(point[1])),
                    20,
                    (0, 255, 0),
                    4,
                )

        if (
            self.analyser.border_vector is not None
            and self.analyser.robot_pos is not None
        ):
            border_vector_end = self.analyser.robot_pos + self.analyser.border_vector
            robot_pos = self.analyser.robot_pos.astype(int)
            border_vector_end = border_vector_end.astype(int)
            cv2.arrowedLine(
                robot_arrows_on_frame,
                tuple(robot_pos),
                tuple(border_vector_end),
                (255, 255, 0),
                2,
            )

        if (
            self.analyser.robot_vector is not None
            and self.analyser.robot_pos is not None
        ):
            robot_vector_end = self.analyser.robot_pos + self.analyser.robot_vector
            robot_pos = self.analyser.robot_pos.astype(int)
            robot_vector_end = robot_vector_end.astype(int)

            cv2.arrowedLine(
                robot_arrows_on_frame,
                tuple(robot_pos),
                tuple(robot_vector_end),
                (0, 0, 255),
                2,
            )
        if self.analyser.robot_vector_not_translated is not None:
            robot_vector_end = (
                self.analyser.robot_pos_not_translated
                + self.analyser.robot_vector_not_translated
            )
            cv2.arrowedLine(
                robot_arrows_on_frame,
                tuple(self.analyser.robot_pos_not_translated.astype(int)),
                tuple(robot_vector_end.astype(int)),
                (0, 255, 0),
                2,
            )

        if (
            self.steering_instance.steering_vector is not None
            and self.analyser.robot_pos is not None
        ):
            ball_vector_end = (
                self.analyser.robot_pos + self.steering_instance.steering_vector
            )
            robot_pos = self.analyser.robot_pos.astype(int)
            ball_vector_end = ball_vector_end.astype(int)

            print(f"ball_vector_end: {ball_vector_end}")
            cv2.arrowedLine(
                robot_arrows_on_frame,
                tuple(robot_pos),
                tuple(ball_vector_end),
                (255, 0, 255),
                2,
            )

        # yellow, blue, green,red
        colors = [
            (0, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
        ]  # Define your colors here

        if self.analyser.corners is not None:
            for index, corner in enumerate(self.analyser.corners):
                color = colors[
                    index % len(colors)
                ]  # This will cycle through the colors

                cv2.circle(frame, tuple(corner), 5, color, -1)
                # Add text above the corner
                text_position = (
                    corner[0],
                    corner[1] - 10,
                )  # Position the text 10 pixels above the corner
                cv2.putText(
                    frame,
                    f"Corner {index + 1}",
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
        else:
            print("No corners found")

        if (
            self.analyser.small_goal_coords is not None
            and self.analyser.large_goal_coords is not None
        ):

            cv2.circle(
                frame,
                tuple(self.analyser.small_goal_coords.astype(int)),
                5,
                (0, 255, 0),
                -1,
            )
            cv2.circle(
                frame,
                tuple(self.analyser.large_goal_coords.astype(int)),
                5,
                (0, 255, 255),
                -1,
            )
            if self.analyser.safepoint_list is None:
                print("Error: safepoint_list is None")
            else:
                for index, coord in enumerate(self.analyser.safepoint_list):
                    color = (0, 255, 255)

                    cv2.circle(frame, tuple(coord.astype(int)), 5, color, -1)
                    # Add text above the corner
                    text_position = (
                        coord[0].astype(int),
                        coord[1].astype(int)- 10,
                    )  # Position the text 10 pixels above the corner
                    cv2.putText(
                        frame,
                        f"Safe  {index}",
                        text_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )


        if self.analyser.translation_vector is not None:
            cv2.arrowedLine(
                frame,
                self.analyser.large_goal_coords.astype(int)
                + self.analyser.translation_vector.astype(int),
                self.analyser.small_goal_coords.astype(int),
                (255, 0, 0),
                2,
            )

        if (
            self.analyser.dropoff_coords is not None
            and self.analyser.robot_pos is not None
        ):
            cv2.circle(
                frame, self.analyser.dropoff_coords.astype(int), 10, (255, 0, 255), -1
            )
        if (
            self.steering_instance.state is not None
            and self.steering_instance.state.path is not None
        ):
            robot_pos = self.analyser.robot_pos.astype(int)
            for idx, point in enumerate(self.steering_instance.state.path):

                # Draw arrows between safepoints
                if idx == 0:
                    cv2.arrowedLine(
                        frame,
                        robot_pos,
                        (point).astype(int),
                        (255, 0, 0),
                        2,
                    )
                else:
                    cv2.arrowedLine(
                        frame,
                        (self.steering_instance.state.path[idx-1]).astype(int),
                        (point).astype(int),
                        (255, 0, 0),
                        2,
                    )

        if self.analyser.middle_point is not None:
            cv2.circle(
                frame, self.analyser.middle_point.astype(int), 30, (255, 0, 0), 0
            )

        self.videoDebugger.write_video("result", result_3channel, True)

        for r in cross_rect:
            cv2.rectangle(
                result_3channel,
                (r[0], r[1]),
                (r[0] + r[2], r[1] + r[3]),
                (0, 0, 255),
                2,
            )
        im1 = cv2.resize(robot_arrows_on_frame, (width, height))
        im2 = cv2.resize(result_3channel, (width, height))
        im3 = cv2.resize(text_overview, (width, height))
        im3 = cv2.resize(border_mask, (width, height))
        im4 = cv2.resize(green_robot_3channel, (width, height))

        hstack1 = np.hstack((im1, im2))
        hstack2 = np.hstack((im3, im4))
        combined_images = np.vstack((hstack1, hstack2))
        self.videoDebugger.write_video("combined_images", combined_images, True)
        cv2.namedWindow("frame", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("frame", combined_images)
