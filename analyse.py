from typing import Tuple, List, Dict
import cv2
import numpy as np
import VideoDebugger
import BlobDetector

class Analyse:
    def __init__(self):
        self.videoDebugger = VideoDebugger.VideoDebugger()
        self.red_pos = None
        self.robot_pos = None
        self.robot_vector = None
        self.corners = None
        self.ball_vector = None
        self.bounds_dict = read_bounds()
        pass
    
    def analysis_pipeline(self, image: np.ndarray, bounds_dict : Dict[str,np.ndarray] ):
        self.videoDebugger.write_video("original", image, True)
        self.green_robot_mask = self.videoDebugger.run_analysis(self.isolate_green_robot, "green-mask", image, lower=bounds_dict["green_lower"], upper=bounds_dict["green_upper"])
        self.red_robot_mask = self.videoDebugger.run_analysis(self.isolate_red_robot, "red-mask", image, lower=bounds_dict["red_lower"], upper=bounds_dict["red_upper"])
        self.border_mask = self.videoDebugger.run_analysis(self.isolate_borders, "border", image, lower=bounds_dict["border_lower"], upper=bounds_dict["border_upper"])
        self.white_mask = self.videoDebugger.run_analysis(self.isolate_white_ball, "white-ball", image, lower=bounds_dict["white_lower"], upper=bounds_dict["white_upper"])
        self.orange_mask = self.videoDebugger.run_analysis(self.isolate_orange_ball, "orange-ball", image, lower=bounds_dict["orange_lower"], upper=bounds_dict["orange_upper"])

        self.white_ball_keypoints = self.find_ball_keypoints(self.white_mask)
        self.orange_ball_keypoints = self.find_ball_keypoints(self.orange_mask)
        self.keypoints = self.white_ball_keypoints + self.orange_ball_keypoints

        try:
            self.robot_pos, self.red_pos, self.robot_vector = self.find_circle_robot(self.green_robot_mask, self.red_robot_mask)
            self.corners = self.find_border_corners(self.border_mask)
            self.ball_vector = self.find_ball_vector(self.white_ball_keypoints, self.robot_pos)
        except BorderNotFoundError as e:
            print(e)
        except RobotNotFoundError as e:
            print(e)
        except Exception as e:
            print(e)
        return

    def isolate_green_robot(self, image: np.ndarray, lower : np.ndarray, upper: np.ndarray) -> np.ndarray:
        # Isolate the green robot
        #lower = np.array([0, 160, 0])
        #upper = np.array([220, 255, 220])
        mask = cv2.inRange(image, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=2)
        return mask

    def isolate_red_robot(self, image: np.ndarray, lower: np.ndarray,upper: np.ndarray)-> np.ndarray:
        # Isolate the red robot
        #lower = np.array([0, 0, 160])
        #upper = np.array([220, 220, 255])
        mask = cv2.inRange(image, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=2)
        return mask
    
    def isolate_orange_ball(self, image: np.ndarray, lower: np.ndarray,upper: np.ndarray)-> np.ndarray:
        #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #lower = np.array([11, 120, 120])
        #upper = np.array([30, 255, 255])
        mask = cv2.inRange(image, lower, upper)
        res = cv2.bitwise_and(image, image, mask=mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=2)
        return mask


    def isolate_white_ball(self, image: np.ndarray,lower: np.ndarray,upper: np.ndarray)-> np.ndarray:
        # turn into hsl
        #hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        #lower = np.array([0, 190, 0])
        #upper = np.array([255, 255, 255])
        mask = cv2.inRange(image, lower, upper)
        res = cv2.bitwise_and(image, image, mask=mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=2)
        return mask



    def find_circle_robot(self, green_mask: np.ndarray, red_mask: np.ndarray) -> Tuple[np.ndarray,np.ndarray, np.ndarray]:
        detector = BlobDetector.get_robot_circle_detector()
        green_keypoints = detector.detect(green_mask)
        red_mask = detector.detect(red_mask)
        print(f"There are {len(green_keypoints)} green points, there should be 1")
        print(f"There are {len(red_mask)} red points, there should be 1")
        if len(green_keypoints) != 1:
            raise RobotNotFoundError(f"Cannot find robot: There are {len(green_keypoints)} green points. There are {len(red_mask)} red points")
        if len(red_mask) != 1:
            raise RobotNotFoundError(f"Cannot find robot: There are {len(green_keypoints)} green points. There are {len(red_mask)} red points")
        print(f"Green found at: {green_keypoints[0].pt}")
        print(f"Red found at: {red_mask[0].pt}")

        green_point = self.convert_perspective(green_keypoints[0].pt)
        red_point = self.convert_perspective(red_mask[0].pt)

        print(f"Green converted to: {green_point}")
        print(f"Red converted to: {red_point}")

        return np.array(green_point), np.array(red_point), self.construct_vector_from_circles(np.array(green_point), np.array(red_point))

    def convert_perspective(self, point : np.ndarray) -> tuple[float, float]:
        # Heights in cm
        cam_height = 200
        robot_height = 40

        # Heights in pixels cm / px
        conversionFactor = 180 / (1920*5/6)

        vector_from_middle = np.array([point[0] - 1920/2, point[1] - 1080/2])
        # Convert to cm
        vector_from_middle *= conversionFactor
        projected_vector = vector_from_middle/cam_height * (cam_height - robot_height)

        # Convert back to pixels
        projected_vector /= conversionFactor

        result = (projected_vector[0] + 1920/2, projected_vector[1] + 1080/2)
        return result

    def construct_vector_from_circles(self, green: np.ndarray, red: np.ndarray) -> np.ndarray:
        return red - green
    
    def isolate_borders(self, image: np.ndarray,lower: np.ndarray,upper: np.ndarray) -> np.ndarray:
        res = image
        # exagregate the difference between red/orange colors
        #hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
        #lower = np.array([0, 80, 140])
        #upper = np.array([13, 255, 255])
        mask = cv2.inRange(image, lower, upper)
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

        #h, w = mask.shape[:2]
        #mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
        #mask[0, :] = 0  # Set top row to black
        #mask[:, 0] = 0  # Set left column to black
        #mask = cv2.floodFill(mask, None, (0, 0), 0, flags=8)[1][1: h + 1, 1: w + 1]
        #mask = cv2.bitwise_not(mask)

        # need to find a better denoise method
        return cv2.bitwise_not(result)

    def find_ball_vector(self, keypoints : np.ndarray, robot_pos : np.ndarray) -> np.ndarray:
        if len(keypoints) == 0:
            raise BallNotFoundError("No balls to be used for vector calculation")
        if robot_pos is None:
            raise RobotNotFoundError("No Robot to be used for vector calculation")
        # Find the closest ball to the robot
        closest_ball = None
        closest_distance = None
        for keypoint in keypoints:
            ball_pos = np.array(keypoint.pt)
            distance = np.linalg.norm(ball_pos - robot_pos)
            if closest_distance is None or distance < closest_distance:
                closest_ball = ball_pos
                closest_distance = distance

        return closest_ball - robot_pos

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
                #cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
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
    
    def read_bounds():
        bounds_dict = {}
        with open("bounds.txt") as f:
            for line in f:
                key, value = line.split(";")
                bounds = value.split(",")
                bounds_dict[key] = np.array([float(x) for x in bounds])
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

#    def find_triangle_robot(self, mask: np.ndarray):
#        # Find the robot in the mask
#        params = cv2.SimpleBlobDetector_Params()
#        params.minThreshold = 1
#        params.maxThreshold = 256
#        params.filterByColor = True
#        params.blobColor = 255
#        params.filterByArea = True
#        params.minArea = 500
#        params.maxArea = 10000000
#        params.filterByCircularity = False
#        params.filterByConvexity = True
#        params.minConvexity = 0.8
#        params.filterByInertia = False
#
#        # Create a detector with the parameters
#        detector = cv2.SimpleBlobDetector_create(params)
#        keypoints = detector.detect(mask)
#        if len(keypoints) != 1:
#            print("Error: robot not found")
#            print(f"Number of keypoints found: {len(keypoints)}")
#            return None, None
#        robot_pos = np.array(keypoints[0].pt)
#        print(f"Robot found at:  {keypoints[0].pt}")
#        # Find contours in the mask
#        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#        # Convert keypoint coordinates to integer
#        x, y = int(keypoints[0].pt[0]), int(keypoints[0].pt[1])
#
#        # Contour that contains the keypoint
#        target_contour = None
#        for contour in contours:
#            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
#                target_contour = contour
#                break
#
#        if target_contour is None:
#            print("Error: Contour for the robot not found")
#            return None, None
#
#        # Create a blank image and draw the target contour filled with white
#        blob_image = np.zeros_like(mask)
#        cv2.drawContours(blob_image, [target_contour], -1, (255), thickness=cv2.FILLED)
#
#        # Optionally, crop the image to the bounding box of the contour to reduce size
#        x, y, w, h = cv2.boundingRect(target_contour)
#        cropped_blob_image = blob_image[y:y + h, x:x + w]
#
#        # Save or process the blob image
#        cv2.imwrite("blob_image.jpg", cropped_blob_image)
#        corners = self.find_robot_corners(blob_image)
#
#        return corners, robot_pos


#    def find_robot_corners(self, mask: np.ndarray):
#        gray = np.float32(mask)
#        dst = cv2.cornerHarris(gray, 8, 5, 0.04)
#        ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
#        dst = np.uint8(dst)
#        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
#        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
#        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
#        print("Robot corners found at:")
#        for i in range(0, len(corners)):
#            print(corners[i])
#
#        if len(corners) != 3:
#            print("Error: robot corners not found")
#            print(f"Number of robot corners found: {len(corners)}")
#            return corners[1:]
#        return corners


#    def construct_vector_from_corners(self, corners: np.ndarray):
#
#        if corners is None:
#            return None
#        if len(corners) != 3:
#            print("Error: not right amount of corners found to construct vector")
#            return None
#
#        # find base of triangle formed by the corners, by finding the two closest corners
#
#        # find the distance between each pair of corners
#        distances = []
#        for i in range(0, 3):
#            for j in range(i + 1, 3):
#                distances.append((i, j, np.linalg.norm(corners[i] - corners[j])))
#        # sort the distances
#        distances.sort(key=lambda x: x[2])
#        # find the two closest corners
#        base = [corners[distances[0][0]], corners[distances[0][1]]]
#        # find midpoint of base
#        base_midpoint = (base[0] + base[1]) / 2
#        # find the third corner
#        top = None
#        for i in range(0, 3):
#            if i not in distances[0][:2]:
#                top = corners[i]
#                break
#        # find the vector from the base midpoint to the top
#        vector = np.array(top - base_midpoint)
#        return vector


