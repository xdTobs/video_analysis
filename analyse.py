from typing import Tuple, List, Dict
import cv2
import numpy as np
import VideoDebugger
import BlobDetector


class Analyse:
    def __init__(self):
        self.videoDebugger = VideoDebugger.VideoDebugger()
        self.alpha = 0.05
        self.robot_pos = None
        self.robot_pos_not_translated = None
        self.robot_vector_not_translated = None
        self.robot_vector = None
        self.corners = None
        self.green_points_not_translated = None
        self.white_average = np.zeros((540, 960), dtype=np.float32)
        self.white_mask = np.zeros((540, 960), dtype=np.float32)
        self.new_white_mask = None
        self.bounds_dict = read_bounds()
        self.distance_to_closest_border = float("inf")
        pass

    def analysis_pipeline(self, image: np.ndarray):

        self.videoDebugger.write_video("original", image, True)
        self.green_robot_mask = self.videoDebugger.run_analysis(
            self.apply_theshold, "green-mask", image, self.bounds_dict["green"]
        )
        self.red_robot_mask = self.videoDebugger.run_analysis(
            self.apply_theshold, "red-mask", image, self.bounds_dict["red"]
        )
        self.border_mask = self.videoDebugger.run_analysis(
            self.isolate_borders, "border", image, self.bounds_dict["border"]
        )
        
        self.new_white_mask = self.videoDebugger.run_analysis(
            self.apply_theshold, "white-ball", image, self.bounds_dict["white"]
        )
        print(image.shape)
        self.white_average = self.alpha * self.new_white_mask + (1 - self.alpha) * self.white_average
        self.white_mask = (self.white_average > 0.5).astype(np.uint8)
        
        self.orange_mask = self.videoDebugger.run_analysis(
            self.apply_theshold, "orange-ball", image, self.bounds_dict["orange"]
        )
        
        self.white_ball_keypoints = self.find_ball_keypoints(self.white_mask)
        self.orange_ball_keypoints = self.find_ball_keypoints(self.orange_mask)
        self.keypoints = self.white_ball_keypoints + self.orange_ball_keypoints
        try:
            self.robot_pos, self.robot_vector = self.find_triple_green_robot(
                self.green_robot_mask
            )
            self.corners = self.find_border_corners(self.border_mask)
            self.distance_to_border = self.distance_to_closest_border()
        except BorderNotFoundError as e:
            print(e)
        except RobotNotFoundError as e:
            print(e)
        except Exception as e:
            print(e)
        return

    def apply_theshold(
        self, image: np.ndarray, bounds_dict_entry: np.ndarray
    ) -> np.ndarray:
        # Isolate the green robot
        # lower = np.array([0, 160, 0])
        # upper = np.array([220, 255, 220])
        # print(f"Bounds dict entry {bounds_dict_entry}")
        bounds = bounds_dict_entry[0:3]
        variance = bounds_dict_entry[3]

        lower = np.clip(bounds - variance, 0, 255)
        upper = np.clip(bounds + variance, 0, 255)
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
        self.green_points_not_translated = [np.array(keypoint.pt) for keypoint in green_keypoints]
        parings = []
        for i in range(0, 3):
            for j in range(i + 1, 3):
                parings.append(
                    (i, j, np.linalg.norm(self.green_points_not_translated[i] - self.green_points_not_translated[j]))
                )
        parings.sort(key=lambda x: x[2])
        # print(f"Parings: {parings}")
        bottom_points = [parings[0][0], parings[0][1]]
        top_point = 3 - bottom_points[0] - bottom_points[1]
        bottom_pos = np.array(
            self.convert_perspective(
                (self.green_points_not_translated[bottom_points[0]] + self.green_points_not_translated[bottom_points[1]]) / 2
            )
        )
        # print(f"Bottom points: {bottom_points}")
        # print(f"Top point: {top_point}")
        top_pos = np.array(self.convert_perspective(self.green_points_not_translated[top_point]))
        # print(f"Bottom pos: {bottom_pos}")
        # print(f"Top pos: {top_pos}")
        self.robot_vector_not_translated = np.array(self.green_points_not_translated[top_point]) - np.array(self.green_points_not_translated[bottom_points[0]] + self.green_points_not_translated[bottom_points[1]])/2
        self.robot_pos_not_translated = (self.green_points_not_translated[bottom_points[0]] + self.green_points_not_translated[bottom_points[1]]) / 2
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

    def convert_perspective(self, point: np.ndarray) -> tuple[float, float]:
        # Heights in cm
        cam_height = 178
        robot_height = 47

        # Heights in pixels cm / px
        conversionFactor = 180 / (1920 / 2 * 5 / 6)

        vector_from_middle = np.array(
            [point[0] - 1920 / 2 / 2, point[1] - 1080 / 2 / 2]
        )
        # Convert to cm
        vector_from_middle *= conversionFactor
        projected_vector = vector_from_middle / cam_height * (cam_height - robot_height)

        # Convert back to pixels
        projected_vector /= conversionFactor

        result = (
            projected_vector[0] + 1920 / 2 / 2,
            projected_vector[1] + 1080 / 2 / 2,
        )
        return result

    def construct_vector_from_circles(
        self, green: np.ndarray, red: np.ndarray
    ) -> np.ndarray:
        return red - green

    def isolate_borders(
        self, image: np.ndarray, bounds_dict_entry: np.ndarray
    ) -> np.ndarray:
        res = image
        # exagregate the difference between red/orange colors
        # hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
        # lower = np.array([0, 80, 140])
        # upper = np.array([13, 255, 255])

        mask = self.apply_theshold(image, bounds_dict_entry)
        res = cv2.bitwise_and(res, res, mask=mask)
        mask = cv2.bitwise_not(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assuming the largest contour is the square
        square_contour = max(contours, key=cv2.contourArea)

        # Create an all black mask
        black_mask = np.zeros_like(mask)

        # Fill the mask with white where the square is
        cv2.drawContours(black_mask, [square_contour], -1, (255), thickness=cv2.FILLED)

        # Apply the mask to the binary image
        result = cv2.bitwise_and(mask, black_mask)
        # flood fill black all white that are touching edge of images

        # h, w = mask.shape[:2]
        # mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
        # mask[0, :] = 0  # Set top row to black
        # mask[:, 0] = 0  # Set left column to black
        # mask = cv2.floodFill(mask, None, (0, 0), 0, flags=8)[1][1: h + 1, 1: w + 1]
        # mask = cv2.bitwise_not(mask)

        # need to find a better denoise method
        return cv2.bitwise_not(result)

    def find_border_corners(self, image: np.ndarray) -> np.ndarray:
        image = cv2.bitwise_not(image)
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

        # cv2.imwrite(os.path.join("./output/", "keypoints.jpg"), res)

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
        return np.linalg.norm(p - projection)

    def distance_to_closest_border(self) -> float:
        if self.robot_pos is None or self.corners is None:
            raise ValueError("Robot position or border corners are not set.")

        num_corners = len(self.corners)
        min_distance = float("inf")
        for i in range(num_corners):
            v = self.corners[i]
            w = self.corners[(i + 1) % num_corners]
            distance = self.distance_point_to_segment(self.robot_pos, v, w)
            if distance < min_distance:
                min_distance = distance

        return min_distance


def read_bounds():
    bounds_dict = {}
    with open("bounds.txt") as f:
        for line in f:
            key, value = line.split(";")
            bounds = value.split(",")
            bounds_dict[key] = np.array([int(x) for x in bounds])
    return bounds_dict

def read_goal():
    goal_coordinates = np.array([])
    with open("bounds.txt") as f:
        for line in f:
            key, value = line.split(";")
            bounds = value.split(",")
            if key == "goal":
                goal_coordinates.append([int(x) for x in bounds])
        return goal_coordinates


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
