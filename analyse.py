from typing import Tuple
import cv2
import numpy as np
import VideoDebugger
import BlobDetector
import traceback
from utils import coordinates_to_vector, angle_between_vectors
import math
import sys
from collections import deque


class Analyse:
    def __init__(self):
        self.videoDebugger = VideoDebugger.VideoDebugger()
        self.alpha = 0.1
        self.average_threshold = 100
        self.robot_pos = None
        self.robot_pos_not_translated = None
        self.robot_vector_not_translated = None
        self.robot_vector = None
        self.goal_vector = None
        self.robot_pos_at_path_creation = None
        self.path = []
        self.path_indexes = []
        self.delivery_vector = None
        self.corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
        self.border_vector = None
        self.small_goal_coords: np.ndarray = None
        self.large_goal_coords: np.ndarray = None
        self.course_length_px = None
        self.course_height_px = None
        self.goal_vector = None
        self.delivery_vector = None
        self.translation_vector = None
        self.green_points_not_translated = None
        self.dropoff_coords = None
        self.should_calculate_corners = True
        self.safepoint_list: np.ndarray = None
        self.middle_point = None
        self.distance_to_middle = None
        self.is_ball_close_to_middle = False

        self.new_white_mask = None
        self.white_average = np.zeros((576, 1024), dtype=np.float32)
        self.white_mask = np.zeros((576, 1024), dtype=np.float32)

        self.border_average = np.zeros((576, 1024), dtype=np.float32)
        self.border_mask = np.zeros((576, 1024), dtype=np.float32)
        self.new_border_mask = None

        self.new_orange_mask = None
        self.orange_average = np.zeros((576, 1024), dtype=np.float32)
        self.orange_mask = np.zeros((576, 1024), dtype=np.float32)

        self.distance_to_closest_border = float("inf")

        self.cam_height = 220
        self.robot_height = 47
        self.course_length_cm = 167
        self.course_width_cm = 121

        pass

    def analysis_pipeline(self, image: np.ndarray, has_found_corners):

        self.videoDebugger.write_video("original", image, True)
        self.green_robot_mask = self.videoDebugger.run_analysis(
            self.apply_threshold, "green-mask", image
        )
        self.red_robot_mask = self.videoDebugger.run_analysis(
            self.apply_threshold, "red-mask", image
        )

        self.new_white_mask = self.videoDebugger.run_analysis(
            self.apply_threshold, "white-ball", image
        )
        self.white_average = (
            self.alpha * self.new_white_mask + (1 - self.alpha) * self.white_average
        )
        self.white_mask = (
            self.white_average.astype(np.uint8) > self.average_threshold
        ).astype(np.uint8) * 255

        self.white_average = (
            self.alpha * self.new_white_mask + (1 - self.alpha) * self.white_average
        )
        self.white_mask = (
            self.white_average.astype(np.uint8) > self.average_threshold
        ).astype(np.uint8) * 255

        self.new_orange_mask = self.videoDebugger.run_analysis(
            self.apply_threshold, "orange-ball", image
        )
        self.orange_average = (
            self.alpha * self.new_orange_mask + (1 - self.alpha) * self.orange_average
        )
        self.orange_mask = (
            self.orange_average.astype(np.uint8) > self.average_threshold
        ).astype(np.uint8) * 255

        self.new_border_mask = self.videoDebugger.run_analysis(
            self.isolate_borders, "border", image
        )
        self.border_average = (
            self.alpha * self.new_border_mask + (1 - self.alpha) * self.border_average
        )
        self.border_mask = (
            self.border_average.astype(np.uint8) > self.average_threshold
        ).astype(np.uint8) * 255

        self.white_ball_keypoints = self.find_ball_keypoints(self.white_mask)
        self.orange_ball_keypoints = self.find_ball_keypoints(self.orange_mask)
        self.keypoints = self.white_ball_keypoints + self.orange_ball_keypoints
        try:
            if not has_found_corners:
                self.corners = self.find_border_corners(self.border_mask)
            self.calculate_goals()
            self.calculate_course_dimensions()
            self.distance_to_closest_border, self.border_vector = (
                self.calculate_distance_to_closest_border(self.robot_pos)
            )
            self.calculate_safepoints()

        except BorderNotFoundError as e:
            print(e)

        except Exception as e:
            traceback.print_exc()

        try:
            self.robot_pos, self.robot_vector = self.find_triple_green_robot(
                self.green_robot_mask
            )
        except RobotNotFoundError as e:
            print(e)
        return

    def calculate_course_dimensions(self):
        if self.corners is not None:
            corner1 = self.corners[0]
            corner2 = self.corners[1]
            corner3 = self.corners[2]
            self.course_length_px = np.linalg.norm(corner1 - corner2)
            self.course_height_px = np.linalg.norm(corner2 - corner3)
            self.middle_point = (corner1 + corner3) / 2

            cross_rect, _ = self.find_cross_bounding_rectangle(self.border_mask)
            if len(cross_rect) > 0:
                self.middle_point = np.array(
                    [
                        cross_rect[0][0] + cross_rect[0][2] // 2,
                        cross_rect[0][1] + cross_rect[0][3] // 2,
                    ]
                )
                # print(f"Cross found at {self.middle_point}")
            self.distance_to_middle = np.linalg.norm(self.robot_pos - self.middle_point)

    def calculate_is_ball_close_to_middle(self, ball_position: np.ndarray, threshold: float = 100) -> bool:
        if self.middle_point is None:
            raise ValueError("Middle point has not been calculated.")

        distance_to_middle = np.linalg.norm(ball_position - self.middle_point)
        return distance_to_middle < threshold

    @staticmethod
    def apply_threshold(image: np.ndarray, out_name: str) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if out_name == "white-ball":
            # https://stackoverflow.com/questions/22588146/tracking-white-color-using-python-opencv
            sensitivity = 35
            lower = np.array([0, 0, 255 - sensitivity])
            upper = np.array([180, sensitivity, 255])
        elif out_name == "green-mask":
            lower = np.array([31, 20, 180])
            upper = np.array([120, 255, 255])
        elif out_name == "red-mask":
            lower = np.array([0, 70, 50])
            upper = np.array([10, 255, 255])
        elif out_name == "orange-ball":
            lower = np.array([10, 5, 220])
            upper = np.array([30, 255, 255])
        elif out_name == "border":
            lower = np.array([0, 150, 100])
            upper = np.array([30, 245, 245])

        mask = cv2.inRange(hsv, lower, upper)

        return mask

    def get_speed(self, distance: int):
        speed = (
            0.01100000000 * math.pow(distance, 2) - 0.1200000000 * distance + 0.1
        ) / 5
        print(f" Distance: {distance}, Speed {speed}")
        return speed

    def are_coordinates_close(self, vector: np.ndarray, dist=100) -> bool:
        length = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
        return length < dist

    def is_point_close(self, point: np.ndarray) -> bool:
        distance = np.linalg.norm(point - self.robot_pos)
        return distance < 100

    def can_target_ball_directly(
        self, robot_pos: np.ndarray, ball_pos: np.ndarray
    ) -> bool:
        distance_to_ball = np.linalg.norm(ball_pos - robot_pos)
        vector_to_ball = ball_pos - robot_pos
        angle_to_ball = math.degrees(
            angle_between_vectors(vector_to_ball, self.robot_vector)
        )
        if distance_to_ball < 250 and angle_to_ball < 80:
            return True
        return False

    def find_steering_vector(
        self,
        robot_pos: np.ndarray,
        target_position: np.ndarray,
    ) -> np.ndarray:

        return target_position - robot_pos

    def create_path(self):
        ball_position = self.find_closest_ball(self.keypoints, self.robot_pos)
        if ball_position is None:
            ball_position = self.dropoff_coords

        self.is_ball_close_to_middle = self.calculate_is_ball_close_to_middle(ball_position)
        if self.is_ball_close_to_middle:
            #TODO: Add logic to handle ball close to middle
            print("Ball is close to middle")



        self.path_indexes = self.find_path_to_target(ball_position, self.robot_pos, self.safepoint_list)
        if len(self.path_indexes) == 0:
            return None
        path = []
        for i in range(0, len(self.path_indexes)):
            steering_vector = self.find_steering_vector(
                self.robot_pos, self.safepoint_list[self.path_indexes[i]]
            )
            # print(f"Index: {i}   Steering vector: {steering_vector}")
            path.append(steering_vector + self.robot_pos)
        steering_vector = self.find_steering_vector(self.robot_pos, ball_position)
        path.append(steering_vector + self.robot_pos)
        self.path = path
        return path

    def find_path_to_target(
        self,
        ball_position: np.ndarray,
        robot_pos: np.ndarray,
        safepoint_list: np.ndarray,
    ) -> np.ndarray:
        closest_safepoint_index_to_ball = self.find_closest_safepoint_index(
            ball_position, safepoint_list
        )
        closest_safepoint_index_to_robot = self.find_closest_safepoint_index(
            robot_pos, safepoint_list
        )

        if closest_safepoint_index_to_robot == closest_safepoint_index_to_ball:
            return [closest_safepoint_index_to_robot]

        queue = deque(
            [(closest_safepoint_index_to_robot, [closest_safepoint_index_to_robot])]
        )
        visited = set()
        visited.add(closest_safepoint_index_to_robot)
        safepoint_count = len(safepoint_list)

        while queue:
            current_index, path = queue.popleft()

            for neighbor in [
                (current_index - 1) % safepoint_count,
                (current_index + 1) % safepoint_count,
            ]:
                if neighbor not in visited:
                    if neighbor == closest_safepoint_index_to_ball:
                        return path + [neighbor]
                    queue.append((neighbor, path + [neighbor]))
                    visited.add(neighbor)
        return []

    def find_closest_ball(
        self, keypoints: np.ndarray, robot_pos: np.ndarray
    ) -> np.ndarray:
        if len(keypoints) == 0:
            return None
        if robot_pos is None:
            return None
        closest_distance = sys.maxsize
        closest_point = None
        for keypoint in keypoints:
            ball_pos = np.array(keypoint.pt)
            distance = np.linalg.norm(ball_pos - robot_pos)
            if distance < closest_distance:
                closest_distance = distance
                closest_point = ball_pos
        return closest_point

    def calculate_is_ball_close_to_borders(
        self, ball_pos: np.ndarray, corners: np.ndarray
    ) -> bool:
        x_min, y_min = np.min(corners, axis=0)
        x_max, y_max = np.max(corners, axis=0)

        x, y = ball_pos
        distance_to_left_border = x - x_min
        distance_to_right_border = x_max - x
        distance_to_bottom_border = y - y_min
        distance_to_top_border = y_max - y

        if (
            distance_to_left_border < self.distance_to_border_threshold
            or distance_to_right_border < self.distance_to_border_threshold
            or distance_to_bottom_border < self.distance_to_border_threshold
            or distance_to_top_border < self.distance_to_border_threshold
        ):
            return True
        return False

    def find_closest_safepoint_index(
        self, position: np.ndarray, safepoint_list: np.ndarray
    ) -> int:
        if len(safepoint_list) == 0:
            return None
        closest_distance = sys.maxsize
        closest_index = 0
        for i, point in enumerate(safepoint_list):

            distance = np.linalg.norm(position - point)
            if distance < closest_distance:
                closest_distance = distance
                closest_index = i
        return closest_index

    def find_triple_green_robot(self, green_mask: np.ndarray):
        # Errode from green mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        green_mask = cv2.erode(green_mask, kernel, iterations=2)
        detector = BlobDetector.get_robot_circle_detector()
        green_keypoints = detector.detect(green_mask)
        if len(green_keypoints) != 3:
            raise RobotNotFoundError(
                f"Cannot find robot: There are {len(green_keypoints)} green points"
            )
        # Find closest pairing of green points
        self.green_points_not_translated = [
            np.array(keypoint.pt) for keypoint in green_keypoints
        ]
        self.green_points_not_translated = [
            np.array(keypoint.pt) for keypoint in green_keypoints
        ]
        parings = []
        for i in range(0, 3):
            for j in range(i + 1, 3):
                parings.append(
                    (
                        i,
                        j,
                        np.linalg.norm(
                            self.green_points_not_translated[i]
                            - self.green_points_not_translated[j]
                        ),
                    )
                )
        parings.sort(key=lambda x: x[2])
        bottom_points = [parings[0][0], parings[0][1]]
        top_point = 3 - bottom_points[0] - bottom_points[1]
        try:
            bottom_pos = np.array(
                self.convert_perspective(
                    (
                        self.green_points_not_translated[bottom_points[0]]
                        + self.green_points_not_translated[bottom_points[1]]
                    )
                    / 2
                )
            )

            top_pos = np.array(
                self.convert_perspective(self.green_points_not_translated[top_point])
            )

        except ValueError as e:
            bottom_pos = np.array((0, 0))
            top_pos = np.array((0, 0))
            print(e)
        self.robot_vector_not_translated = (
            np.array(self.green_points_not_translated[top_point])
            - np.array(
                self.green_points_not_translated[bottom_points[0]]
                + self.green_points_not_translated[bottom_points[1]]
            )
            / 2
        )
        self.robot_pos_not_translated = (
            self.green_points_not_translated[bottom_points[0]]
            + self.green_points_not_translated[bottom_points[1]]
        ) / 2
        return bottom_pos, top_pos - bottom_pos

    def find_red_green_robot(
        self, green_mask: np.ndarray, red_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        detector = BlobDetector.get_robot_circle_detector()
        green_keypoints = detector.detect(green_mask)
        red_mask = detector.detect(red_mask)

        if len(green_keypoints) != 1:
            raise RobotNotFoundError(
                f"Cannot find robot: There are {len(green_keypoints)} green points. There are {len(red_mask)} red points"
            )
        if len(red_mask) != 1:
            raise RobotNotFoundError(
                f"Cannot find robot: There are {len(green_keypoints)} green points. There are {len(red_mask)} red points"
            )

        try:
            green_point = self.convert_perspective(green_keypoints[0].pt)
            red_point = self.convert_perspective(red_mask[0].pt)
        except ValueError as e:
            green_point = (0, 0)
            red_point = (0, 0)
            print(e)

        return (
            np.array(green_point),
            np.array(red_point),
            self.construct_vector_from_circles(
                np.array(green_point), np.array(red_point)
            ),
        )

    def calculate_goals(self):
        if self.corners is not None:
            # Find the middle of the two corners
            goal_side_right = True

            corner1 = self.corners[0]
            corner2 = self.corners[1]
            corner3 = self.corners[2]
            corner4 = self.corners[3]

            if goal_side_right:
                self.small_goal_coords = (corner1 + corner2) // 2
                self.large_goal_coords = (corner3 + corner4) // 2
            else:
                self.small_goal_coords = (corner3 + corner4) // 2
                self.large_goal_coords = (corner1 + corner2) // 2

            self.goal_vector = coordinates_to_vector(
                self.small_goal_coords, self.large_goal_coords
            )

            self.goal_vector = coordinates_to_vector(
                self.large_goal_coords, self.small_goal_coords
            )

            self.translation_vector = self.goal_vector * 0.9

            self.dropoff_coords = self.large_goal_coords + self.translation_vector
            self.delivery_vector = coordinates_to_vector(
                self.dropoff_coords, self.small_goal_coords
            )

    def calculate_safepoints(self):
        if self.corners is not None:
            corner1 = self.corners[0]
            corner2 = self.corners[1]
            corner3 = self.corners[2]
            corner4 = self.corners[3]

            small_goal_coords = self.small_goal_coords
            large_goal_coords = self.large_goal_coords

            right_lower_coords = ((corner1 + small_goal_coords) // 2) - [0, 20]
            right_upper_coords = ((small_goal_coords + corner2) // 2) + [0, 20]
            left_lower_coords = ((corner4 + large_goal_coords) // 2) - [0, 20]
            left_upper_coords = ((large_goal_coords + corner3) // 2) + [0, 20]
            # print("Right lower: ", right_lower_coords)
            # print("Right upper: ", right_upper_coords)
            # print("Left lower: ", left_lower_coords)
            # print("Left upper: ", left_upper_coords)

            lower_vector = coordinates_to_vector(right_lower_coords, left_lower_coords)
            upper_vector = coordinates_to_vector(right_upper_coords, left_upper_coords)

            small_translation_vector = lower_vector * 1 / 6
            large_translation_vector = lower_vector * 0.4

            safe_point_1 = right_lower_coords + small_translation_vector
            safe_point_2 = right_lower_coords + large_translation_vector
            safe_point_3 = left_lower_coords - large_translation_vector
            safe_point_4 = left_lower_coords - small_translation_vector
            safe_point_5 = large_goal_coords - small_translation_vector
            safe_point_6 = left_upper_coords - small_translation_vector
            safe_point_7 = left_upper_coords - large_translation_vector
            safe_point_8 = right_upper_coords + large_translation_vector
            safe_point_9 = right_upper_coords + small_translation_vector
            safe_point_10 = small_goal_coords + small_translation_vector

            self.safepoint_list = np.array(
                [
                    safe_point_1,
                    safe_point_2,
                    safe_point_3,
                    safe_point_4,
                    safe_point_5,
                    safe_point_6,
                    safe_point_7,
                    safe_point_8,
                    safe_point_9,
                    safe_point_10,
                ]
            )
            # print("Safepoints: ", self.safepoint_list)
            return

    def convert_perspective(self, point: np.ndarray) -> tuple[float, float]:
        # Heights in cm
        # print(f"course height and length px {self.course_height_px} {self.course_length_px}")

        # Heights in pixels cm / px
        # TODO fish eye ???
        if self.course_length_px is None:
            raise ValueError("Course length is not set")
        conversionFactor = self.course_length_cm / (
            self.course_length_px * 1024 / self.course_length_cm
        )

        vector_from_middle = np.array([point[0] - 1024 / 2, point[1] - 576 / 2])
        # Convert to cm
        vector_from_middle *= conversionFactor

        projected_vector = (
            vector_from_middle / self.cam_height * (self.cam_height - self.robot_height)
        )

        # Convert back to pixels
        projected_vector /= conversionFactor

        result = (
            projected_vector[0] + 1024 / 2,
            projected_vector[1] + 576 / 2,
        )
        return result

    def construct_vector_from_circles(
        self, green: np.ndarray, red: np.ndarray
    ) -> np.ndarray:
        return red - green

    # returns the x, y, width and height of a rectangle that contains the cross
    @staticmethod
    def find_cross_bounding_rectangle(mask: np.ndarray) -> np.ndarray:
        if len(mask.shape) != 2 or mask.dtype != np.uint8:
            raise ValueError(
                "Input mask must be a single-channel binary image of type uint8"
            )

        h, w = mask.shape[:2]
        flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(mask, flood_fill_mask, (0, 0), 255)

        mask = cv2.bitwise_not(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and detect crosses or plus signs
        mid_cross_rect = []
        approx_contour = None
        for contour in contours:
            # Approximate the contour to reduce the number of points
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the approximated contour has the characteristics of a cross
            if (
                len(approx) >= 4 and cv2.contourArea(contour) > 100
            ):  # Area threshold to filter noise
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                # Check if the contour has an aspect ratio close to 1 (square-like shape)
                if 0.8 < aspect_ratio < 1.2 and w > 30 and h > 30:
                    mid_cross_rect.append((x, y, w, h))
                    approx_contour = approx
        # draw the bounding rectangle on mask
        # mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # for x, y, w, h in mid_cross_rect:
        #     cv2.rectangle(mask_bgr, (x, y), (x + w, y + h), (0, 255, 255), 2)
        # if approx_contour is not None:
        #     cv2.drawContours(mask_bgr, [approx_contour], -1, (0, 255, 0), 2)
        # cv2.imwrite("debug_img/after.jpg", mask_bgr)

        return (mid_cross_rect, approx_contour)

    @staticmethod
    def isolate_borders(image: np.ndarray, out_name) -> np.ndarray:
        mask = Analyse.apply_threshold(image, out_name)
        mask = cv2.bitwise_not(mask)

        h, w = mask.shape[:2]
        mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
        mask = cv2.floodFill(mask, None, (0, 0), 0, flags=8)[1][1 : h + 1, 1 : w + 1]
        return mask

    def find_border_corners(self, image: np.ndarray) -> np.ndarray:
        image = self.border_average.astype(np.uint8)
        contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        corners = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                corners = approx.squeeze()
        if corners is None:
            self.corners

        corner1 = max(corners, key=lambda ci: sum(ci))
        remaining_corners = [c for c in corners if not np.array_equal(c, corner1)]

        corner3 = min(remaining_corners, key=lambda ci: sum(ci))
        remaining_corners = [
            c for c in remaining_corners if not np.array_equal(c, corner3)
        ]
        corner2, corner4 = sorted(remaining_corners, key=lambda ci: ci[1])

        # Replace self.corners with the corners in the correct order
        corners = np.array([corner1, corner2, corner3, corner4])
        if corners is None:
            raise BorderNotFoundError()
        return corners

    def find_ball_keypoints(self, mask: np.ndarray) -> np.ndarray:
        detector = BlobDetector.get_ball_detector()
        keypoints = detector.detect(mask)
        return keypoints
        pass


    def distance_point_to_segment(
        self, p: np.ndarray, v: np.ndarray, w: np.ndarray
    ) -> float:
        l2 = np.sum((w - v) ** 2)
        if l2 == 0.0:
            return np.linalg.norm(p - v)
        t = max(0, min(1, np.dot(p - v, w - v) / l2))
        projection = v + t * (w - v)
        return np.linalg.norm(p - projection), projection - p

    def calculate_distance_to_closest_border(self, pos: np.ndarray) -> float:
        if pos is None or self.corners is None:
            raise ValueError("Position or border corners are not set.")

        num_corners = len(self.corners)
        min_distance = float("inf")
        closest_projection = None
        for i in range(num_corners):
            v = self.corners[i]
            w = self.corners[(i + 1) % num_corners]
            distance, projection_vector = self.distance_point_to_segment(
                self.robot_pos, v, w
            )
            if distance < min_distance:
                min_distance = distance
                closest_projection = projection_vector

        return min_distance, closest_projection

    def angle_between_vectors(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cos_theta = dot_product / (norm_vec1 * norm_vec2)
        angle_radians = np.arccos(
            np.clip(cos_theta, -1.0, 1.0)
        )  # Clip to handle numerical issues
        angle_degrees = np.degrees(angle_radians)
        return angle_radians, angle_degrees

    def get_border_corners(self):
        if self.corners is None:
            raise BorderNotFoundError("No border corners found")
        return self.corners


class RobotNotFoundError(Exception):
    def __init__(self, message="Robot not found", *args):
        super().__init__(message, *args)
        self.message = message


class BallNotFoundError(Exception):
    def __init__(self, message="Ball not found", *args):
        super().__init__(message, *args)
        self.message = message


class BorderNotFoundError(Exception):
    def __init__(self, message="Border not found", *args):
        super().__init__(message, *args)
        self.message = message


class AnalyseError(Exception):
    def __init__(self, message="Failed to analyse image", *args):
        super().__init__(message, *args)
        self.message = message


if __name__ == "__main__":
    img = cv2.imread("bin.jpg")
    binary_img = Analyse.apply_threshold(img, "white-ball")
    bounding_rect, contour = Analyse.find_cross_bounding_rectangle(binary_img)
