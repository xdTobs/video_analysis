from typing import Tuple
import cv2
import numpy as np
import VideoDebugger
import BlobDetector
import traceback


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
        self.delivery_vector = None
        self.corners = None
        self.border_vector = None
        self.small_goal_coords: np.ndarray = None
        self.large_goal_coords: np.ndarray = None
        self.course_length_px = None
        self.course_height_px = None
        self.distance_to_goal = 100
        self.goal_vector = None
        self.delivery_vector = None
        self.translation_vector = None
        self.green_points_not_translated = None
        self.dropoff_coords = None

        self.new_white_mask = None
        self.white_average = np.zeros((576, 1024), dtype=np.float32)
        self.white_mask = np.zeros((576, 1024), dtype=np.float32)

        self.border_average = np.zeros((576, 1024), dtype=np.float32)
        self.border_mask = np.zeros((576, 1024), dtype=np.float32)
        self.new_border_mask = None

        self.new_orange_mask = None
        self.orange_average = np.zeros((576, 1024), dtype=np.float32)
        self.orange_mask = np.zeros((576, 1024), dtype=np.float32)

        self.bounds_dict = read_bounds()
        self.distance_to_closest_border = float("inf")

        self.cam_height = 178
        self.robot_height = 47
        self.course_length_cm = 167
        self.course_width_cm = 121
        pass

    def analysis_pipeline(self, image: np.ndarray):

        self.videoDebugger.write_video("original", image, True)
        self.green_robot_mask = self.videoDebugger.run_analysis(
            self.apply_threshold, "green-mask", image, self.bounds_dict["green"]
        )
        self.red_robot_mask = self.videoDebugger.run_analysis(
            self.apply_threshold, "red-mask", image, self.bounds_dict["red"]
        )

        self.new_white_mask = self.videoDebugger.run_analysis(
            self.apply_threshold, "white-ball", image, self.bounds_dict["white"]
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
            self.apply_threshold, "orange-ball", image, self.bounds_dict["orange"]
        )
        self.orange_average = (
            self.alpha * self.new_orange_mask + (1 - self.alpha) * self.orange_average
        )
        self.orange_mask = (
            self.orange_average.astype(np.uint8) > self.average_threshold
        ).astype(np.uint8) * 255

        self.new_border_mask = self.videoDebugger.run_analysis(
            self.isolate_borders, "border", image, self.bounds_dict["border"]
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
            self.corners = self.find_border_corners(self.border_mask)
            self.calculate_course_dimensions()
            self.calculate_goals()
<<<<<<< HEAD
            self.distance_to_closest_border, self.border_vector = self.calculate_distance_to_closest_border(self.robot_pos)

=======
            self.distance_to_closest_border = self.calculate_distance_to_closest_border(
                self.robot_pos
            )
>>>>>>> 751c565faa43333dd445fb4ca7dadb7e571acf65

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

    @staticmethod
    def apply_threshold(
        image: np.ndarray, bounds_dict_entry: np.ndarray, outname: str
    ) -> np.ndarray:
        bounds = bounds_dict_entry[0:3]
        variance = bounds_dict_entry[3]

        lower = np.clip(bounds - variance, 0, 255)
        upper = np.clip(bounds + variance, 0, 255)

        if outname == "white-ball":
            lower = np.array([0, 0, 200])
            upper = np.array([179, 55, 255])

        # print(lower, upper)
        frame_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame_HSV, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=2)
        return mask

    def find_triple_green_robot(self, green_mask: np.ndarray):
        # Errode from green mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        green_mask = cv2.erode(green_mask, kernel, iterations=2)
        detector = BlobDetector.get_robot_circle_detector()
        green_keypoints = detector.detect(green_mask)
        # print(f"Green points {[green_keypoint.pt for green_keypoint in green_keypoints]}")
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
        # print(f"Parings: {parings}")
        bottom_points = [parings[0][0], parings[0][1]]
        top_point = 3 - bottom_points[0] - bottom_points[1]
        bottom_pos = np.array(
            self.convert_perspective(
                (
                    self.green_points_not_translated[bottom_points[0]]
                    + self.green_points_not_translated[bottom_points[1]]
                )
                / 2
            )
        )
        # print(f"Bottom points: {bottom_points}")
        # print(f"Top point: {top_point}")
        top_pos = np.array(
            self.convert_perspective(self.green_points_not_translated[top_point])
        )
        top_pos = np.array(
            self.convert_perspective(self.green_points_not_translated[top_point])
        )
        # print(f"Bottom pos: {bottom_pos}")
        # print(f"Top pos: {top_pos}")
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
        # print(f"There are {len(green_keypoints)} green points, there should be 1")
        # print(f"There are {len(red_mask)} red points, there should be 1")
        if len(green_keypoints) != 1:
            raise RobotNotFoundError(
                f"Cannot find robot: There are {len(green_keypoints)} green points. There are {len(red_mask)} red points"
            )
        if len(red_mask) != 1:
            raise RobotNotFoundError(
                f"Cannot find robot: There are {len(green_keypoints)} green points. There are {len(red_mask)} red points"
            )
        # print(f"Green found at: {green_keypoints[0].pt}")
        # print(f"Red found at: {red_mask[0].pt}")

        green_point = self.convert_perspective(green_keypoints[0].pt)
        red_point = self.convert_perspective(red_mask[0].pt)

        # print(f"Green converted to: {green_point}")
        # print(f"Red converted to: {red_point}")

        return (
            np.array(green_point),
            np.array(red_point),
            self.construct_vector_from_circles(
                np.array(green_point), np.array(red_point)
            ),
        )

    def calculate_goals(self):
        print(f"corners: {self.corners}")
        if self.corners is not None:
            # Find the middle of the two corners
            goal_side_right = True
            print(f"Goal side right: {goal_side_right}")
            corner1 = self.corners[0]
            corner2 = self.corners[1]
            corner3 = self.corners[2]
            corner4 = self.corners[3]

            if goal_side_right:
                self.small_goal_coords = (corner3 + corner4) / 2
                self.large_goal_coords = (corner1 + corner2) / 2
            else:
                self.small_goal_coords = (corner1 + corner2) / 2
                self.large_goal_coords = (corner3 + corner4) / 2

            print(f"Small goal coords: {self.small_goal_coords}")
            print(f"Large goal coords: {self.large_goal_coords}")

            self.goal_vector = self.coordinates_to_vector(
                self.small_goal_coords, self.large_goal_coords
            )

            self.goal_vector = self.coordinates_to_vector(
                self.large_goal_coords, self.small_goal_coords
            )

            self.translation_vector = self.goal_vector * 4 / 5

            print(f"translation vector: {self.translation_vector}")

            self.delivery_vector = self.coordinates_to_vector(
                self.large_goal_coords + self.translation_vector, self.small_goal_coords
            )

            print(f"delivery vector: {self.delivery_vector}")

    def convert_perspective(self, point: np.ndarray) -> tuple[float, float]:
        # Heights in cm
        print(f"course length px {self.course_height_px} {self.course_length_px}")

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

    def coordinates_to_vector(
        self, point1: float, point2: float
    ) -> np.ndarray[int, int]:
        point1_int = np.array([int(point1[0]), int(point1[1])])
        point2_int = np.array([int(point2[0]), int(point2[1])])
        return point2_int - point1_int

    # returns the x, y, width and height of a rectangle that contains the cross
    @staticmethod
    def find_cross(mask: np.ndarray) -> np.ndarray:
        if len(mask.shape) != 2 or mask.dtype != np.uint8:
            raise ValueError(
                "Input mask must be a single-channel binary image of type uint8"
            )

        h, w = mask.shape[:2]
        flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(mask, flood_fill_mask, (0, 0), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crosses = []
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) >= 4 and cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.8 < aspect_ratio < 1.2:
                    crosses.append((x, y, w, h))

        return crosses

    @staticmethod
    def isolate_borders(
        image: np.ndarray, bounds_dict_entry: np.ndarray, outname
    ) -> np.ndarray:
        mask = Analyse.apply_threshold(image, bounds_dict_entry, outname)
        mask = cv2.bitwise_not(mask)

        h, w = mask.shape[:2]
        mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
        mask = cv2.floodFill(mask, None, (0, 0), 0, flags=8)[1][1 : h + 1, 1 : w + 1]
        return mask

    def find_border_corners(self, image: np.ndarray) -> np.ndarray:
        # image = cv2.bitwise_not(image)
        image = self.border_average.astype(np.uint8)
        contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        corners = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                # cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
                corners = approx.squeeze()
        if corners is None:
            raise BorderNotFoundError()
        return corners

    def find_ball_keypoints(self, mask: np.ndarray) -> np.ndarray:
        # # Find Canny edges
        # edged = cv2.Canny(image, 30, 200)
        # contours, hierarchy = cv2.findContours(
        #     edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        # )
        # for cnt in contours:
        #     print(cnt)
        # Setup SimpleBlobDetector parameters.
        # Threshold image to binary image
        detector = BlobDetector.get_ball_detector()
        keypoints = detector.detect(mask)
        # res = cv2.drawKeypoints(
        #    mask,
        #    keypoints,
        #    np.array([]),
        #    (0, 0, 255),
        #    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        # )

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

        print(f"Distance to closest border: {min_distance}")
        return min_distance, closest_projection
    
    def angle_between_vectors(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cos_theta = dot_product / (norm_vec1 * norm_vec2)
        angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to handle numerical issues
        angle_degrees = np.degrees(angle_radians)
        return angle_radians, angle_degrees


def read_bounds():
    bounds_dict = {}
    with open("bounds.txt") as f:
        for line in f:
            key, value = line.split(";")
            bounds = value.split(",")
            bounds_dict[key] = np.array([int(x) for x in bounds])
    return bounds_dict


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
