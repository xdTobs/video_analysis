import numpy as np
import sys
import math
import time
from analyse import RobotNotFoundError, BorderNotFoundError
import RobotInterface
from utils import angle_between_vectors, angle_between_vectors_signed
from analyse import Analyse
from collections import deque


class Steering:
    def __init__(self, online=True, host="", port=0):
        self.robot_interface = RobotInterface.RobotInterface(host, port, online)
        if online:
            self.robot_interface.connect()
        self.steering_vector = None
        self.is_ball_close_to_border = False
        self.last_target_time = 0
        self.target_ball = None
        self.update_interval = 20  # Time in seconds
        self.distance_threshold_max = 500  # Distance threshold for starting the timer
        self.distance_threshold_min = 100
        self.collect_ball_distance = 150
        self.signed_angle_radians = None
        self.signed_angle_degrees = None
        self.angle_radians = None
        self.angle_degrees = None
        self.close_to_ball = False
        self.current_time = 0
        self.time_to_switch_target = 0
        self.distance_to_border_threshold = 100
        self.distance_to_delivery_point = 30  # The distance where we want to reverse belt and deliver balls
        self.robot_pos = None
        self.robot_vector = None
        self.path = []
        self.is_collecting_balls = True
        # first is if we are turning second is if we are turning right
        self.turn_start = None
        self.target_safepoint_index = None
        self.is_targeting_ball = False
        self.is_targeting_safepoint = False

    # checks if we can go to ball without crashing into the mid cross
    def check_no_obstacles(
        self, robot_pos: np.ndarray, target_pos: np.ndarray, obstacles: np.ndarray
    ) -> bool:
        return True

    def find_steering_vector(
        self,
        robot_pos: np.ndarray,
        target_position: np.ndarray,
    ) -> np.ndarray:

        return target_position - robot_pos

    def find_path_to_target(self, ball_position: np.ndarray, robot_pos: np.ndarray, safepoint_list: np.ndarray) -> np.ndarray:
        closest_safepoint_index_to_ball = self.find_closest_safepoint_index(ball_position, safepoint_list)
        closest_safepoint_index_to_robot = self.find_closest_safepoint_index(robot_pos, safepoint_list)

        if closest_safepoint_index_to_robot == closest_safepoint_index_to_ball:
            return [closest_safepoint_index_to_robot]

        queue = deque([(closest_safepoint_index_to_robot, [closest_safepoint_index_to_robot])])
        visited = set()
        visited.add(closest_safepoint_index_to_robot)
        safepoint_count = len(safepoint_list)

        while queue:
            current_index, path = queue.popleft()

            for neighbor in [(current_index - 1) % safepoint_count, (current_index + 1) % safepoint_count]:
                if neighbor not in visited:
                    if neighbor == closest_safepoint_index_to_ball:
                        return path + [neighbor]
                    queue.append((neighbor, path + [neighbor]))
                    visited.add(neighbor)

        return []

    def create_path(self, ball_position: np.ndarray, robot_pos: np.ndarray, safepoint_list: np.ndarray):
        path_indexes = self.find_path_to_target(ball_position, robot_pos, safepoint_list)
        if len(path_indexes) == 0:
            return None
        path = []
        for i in range (0, len(path_indexes)):
            steering_vector = self.find_steering_vector(robot_pos, safepoint_list[path_indexes[i]])
            print(f"Index: {i}   Steering vector: {steering_vector}")
            path.append(steering_vector)
        steering_vector = self.find_steering_vector(robot_pos, ball_position)
        path.append(steering_vector)
        return path

    def follow_path(self, keypoints: np.ndarray, robot_pos: np.ndarray, safepoint_list: np.ndarray) -> np.ndarray:
        if self.should_switch_target(robot_pos, self.target_ball):
            self.last_target_time = time.time()
            self.target_ball = self.find_closest_ball(keypoints, robot_pos)
        self.path = self.create_path(self.target_ball, robot_pos, safepoint_list)
        if self.are_coordinates_close(self.path[0]) and len(self.path) > 1:
            self.path.pop(0)
        elif self.can_target_ball_directly(robot_pos, self.target_ball):
            while len(self.path) > 1:
                self.path.pop(0)
        self.steering_vector = self.path[0]

    def can_target_ball_directly(self, robot_pos: np.ndarray, ball_pos: np.ndarray) -> bool:
        distance_to_ball = np.linalg.norm(ball_pos - robot_pos)
        if distance_to_ball < 250:
            return True
        return False


    def should_switch_target(self, robot_pos: np.ndarray, ball_pos: np.ndarray) -> bool:
        if ball_pos is None:
            return True
        distance_to_ball = np.linalg.norm(ball_pos - robot_pos)
        if self.is_target_expired() or self.target_ball is None or distance_to_ball < self.distance_threshold_min:
            return True
        return False

    def has_valid_path(self, robot_pos, robot_vector, ball_pos) -> bool:
        return True

    def is_target_expired(self):
        self.current_time = time.time()
        self.time_to_switch_target = self.current_time - self.last_target_time
        if self.time_to_switch_target > self.update_interval:
            self.target_ball = None
            return True
        return False

    def find_closest_ball(self, keypoints: np.ndarray, robot_pos: np.ndarray) -> np.ndarray:
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

    def find_closest_safepoint_index(self, position: np.ndarray, safepoint_list: np.ndarray) -> int:
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

    def are_coordinates_close(self, vector: np.ndarray) -> bool:
        length = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
        print(f"Length: {length}")
        return length < 100


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

    def pick_program_pipeline (
        self,
        keypoints: np.ndarray,
        robot_pos: np.ndarray,
        robot_vector: np.ndarray,
        robot_distance_to_closest_border: float,
        border_vector: np.ndarray,
        corners: np.ndarray,
        dropoff_coords: np.ndarray,
        safepoint_list: np.ndarray,
        border_mask,
    ):

        self.robot_pos = robot_pos
        self.robot_vector = robot_vector

        # if we have a target and no keypoints we still want to catch last ball
        if robot_pos is None:
            raise RobotNotFoundError("No Robot to be used for program selection")
        if robot_vector is None:
            raise RobotNotFoundError("No Robot vector to be used for program selection")
        if robot_distance_to_closest_border is None:
            raise BorderNotFoundError(
                "No distance to closest border to be used for program selection"
            )
        if corners is None:
            raise TypeError("No corners found in pick_program")

        self.follow_path(keypoints, robot_pos, safepoint_list)
        if not self.is_collecting_balls:
            self.deliver_balls_to_target(robot_vector, dropoff_coords, robot_pos)
        if self.steering_vector is None:
            self.is_collecting_balls = False
        else:
            self.is_collecting_balls = True

        if self.is_collecting_balls:
            self.signed_angle_radians = angle_between_vectors_signed(robot_vector, self.steering_vector)  # type: ignore
            self.signed_angle_degrees = math.degrees(self.signed_angle_radians)
            self.angle_radians = angle_between_vectors(robot_vector, self.steering_vector)  # type: ignore
            self.angle_degrees = math.degrees(self.angle_radians)

        dist_to_target = math.sqrt(self.steering_vector[0] ** 2 + self.steering_vector[1] ** 2)
        dist_to_ball = np.linalg.norm(self.target_ball - robot_pos)

        self.is_ball_close_to_border = self.calculate_is_ball_close_to_borders(
            self.target_ball, corners
        )
        print("Steering vector: ", self.steering_vector)
        print(f"Steering vector length: {dist_to_target}")

        try:
            if dist_to_ball < self.collect_ball_distance:
                self.close_to_ball = True
                print("Ball is close")
                self.collect_ball(
                    self.signed_angle_degrees, self.angle_degrees, dist_to_ball
                )
                return

            if dist_to_ball > self.collect_ball_distance:
                print("Ball is not close")
                self.close_to_ball = False
                self.get_near_ball(
                    self.signed_angle_degrees, self.angle_degrees, dist_to_target
                )
                return
            else:
                print("No program picked")

        except ConnectionError as e:
            print(f"Connection error {e}")
            return
        except Exception as e:
            print(f"Error: {e}")
            return


    def move_corrected(self, signed_angle_degrees, angle_degrees, speed):
        print(f"angle to target {angle_degrees}")
        if angle_degrees < 1.5:
            self.robot_interface.send_command("move", 100, speed)
        elif 1.5 <= angle_degrees <= 8:
            self.robot_interface.send_command("move-corrected", -1 * signed_angle_degrees, 40)
            print(f"Signed angle degrees {signed_angle_degrees}")
        elif angle_degrees > 8:
            turn = signed_angle_degrees * -1 / 3
            self.robot_interface.send_command("turn", turn, 30)

    def get_near_ball(self, signed_angle_degrees, angle_degrees, dist_to_ball):
        self.move_corrected(signed_angle_degrees, angle_degrees, 30)

    def collect_ball(self, signed_angle_degrees, angle_degrees, dist_to_ball):
        self.move_corrected(signed_angle_degrees, angle_degrees, 30)

    def start_belt(self):
        self.robot_interface.send_command("belt", 0, speedPercentage=100)

    def stop_belt(self):
        self.robot_interface.send_command("belt", 0, speedPercentage=-100)
        time.sleep(5)
        self.robot_interface.send_command("belt", 0, speedPercentage=0)

    def disconnect(self):
        print("Disconnecting from robot")
        self.robot_interface.disconnect()
        return

    def deliver_balls_to_target(
        self, robot_vector: np.ndarray, dropoff_cords: np.ndarray, robot_pos: np.ndarray
    ):
        if self.drive_to_delivery_point(robot_vector, dropoff_cords, robot_pos):
            self.stop_belt()

    def drive_to_delivery_point(
        self, robot_vector: np.ndarray, target_pos: np.ndarray, robot_pos: np.ndarray
    ):
        # Calculate the vector to the target position
        print(f"Robot pos: {robot_pos}, target pos: {target_pos}\n")
        vector_to_position = target_pos - robot_pos
        distance_to_target = np.linalg.norm(vector_to_position)

        # Normalize the vector to get direction
        direction_to_target = vector_to_position / distance_to_target

        # Calculate the signed angle between the robot's orientation and the target direction
        signed_angle_radians = angle_between_vectors_signed(
            robot_vector, direction_to_target
        )
        self.signed_angle_degrees = math.degrees(signed_angle_radians)
        self.angle_degrees = self.signed_angle_degrees
        print(f"angle to target {self.angle_degrees}")

        if (
            distance_to_target <= self.distance_to_delivery_point
            and self.angle_degrees < 5
        ):
            return True

        if distance_to_target < self.collect_ball_distance:
            print("Delivery point is close")
            self.collect_ball(
                self.signed_angle_degrees, self.angle_degrees, distance_to_target
            )
            return False
        else:
            print("Delivery Point is not close")
            self.get_near_ball(
                self.signed_angle_degrees, self.angle_degrees, distance_to_target
            )
            return False
