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
        self.update_interval = 25  # Time in seconds
        self.distance_threshold_max = 500  # Distance threshold for starting the timer
        self.distance_threshold_min = 100
        self.collect_ball_distance = 150
        self.signed_angle_radians = None
        self.signed_angle_degrees = None
        self.angle_radians = None
        self.angle_degrees = None
        self.close_to_target = False
        self.current_time = 0
        self.time_to_switch_target = 0
        self.distance_to_border_threshold = 100
        self.distance_to_delivery_point = 100  # The distance where we want to reverse belt and deliver balls
        self.robot_pos = None
        self.robot_vector = None
        self.path = []
        self.path_indexes = []
        self.is_collecting_balls = True
        # first is if we are turning second is if we are turning right
        self.turn_start = None
        self.target_safepoint_index = None
        self.is_targeting_ball = False
        self.is_targeting_safepoint = False
        self.speed = 100

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
        self.path_indexes = self.find_path_to_target(ball_position, robot_pos, safepoint_list)
        if not self.is_collecting_balls:
            for i in range(0, len(self.path_indexes)):
                if self.path_indexes[i] == 0 or self.path_indexes[i] == 8:
                    self.path_indexes.pop(i)
                    break
        print(f"Path indexes: {self.path_indexes}")

        if len(self.path_indexes) == 0:
            return None
        path = []
        for i in range (0, len(self.path_indexes)):
            steering_vector = self.find_steering_vector(robot_pos, safepoint_list[self.path_indexes[i]])
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
        if self.is_collecting_balls:
            distance_to_ball = np.linalg.norm(ball_pos - robot_pos)
            if distance_to_ball < 250:
                return True
        return False

    def set_speed(self, distance: int, angle_signed_radians: float):
        angle_radians = abs(angle_signed_radians)
        self.speed = (0.01100000000*math.pow(distance,2) - 0.1200000000 * distance + 0.1)/5

        return self.speed

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
        print(f"Keypoints: {keypoints}")
        if len(keypoints) == 0:
            self.is_collecting_balls = False
            return self.dropoff_coords
        else:
            self.is_collecting_balls = True
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
        if self.is_collecting_balls:
            return length < 40
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
        small_goal_coords: np.ndarray,
        border_mask,

    ):

        self.robot_pos = robot_pos
        self.robot_vector = robot_vector
        self.dropoff_coords = dropoff_coords
        self.small_goal_coords = small_goal_coords

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
            if self.is_ready_to_ejaculate():
                self.ejaculate()



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
                self.close_to_target = True
                print("Target is close")
                self.collect_ball(
                    self.signed_angle_degrees, self.angle_degrees, dist_to_ball
                )
                return

            if dist_to_ball > self.collect_ball_distance:
                print("Target is not close")
                self.close_to_target = False
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


    def move_corrected(self, signed_angle_degrees, angle_degrees):
        print(f"angle to target {angle_degrees}")
        if angle_degrees < 1.5:
            self.robot_interface.send_command("move", 100, self.speed)
        elif 1.5 <= angle_degrees <= 20:
            self.robot_interface.send_command("move-corrected", -1 * signed_angle_degrees, self.speed)
            print(f"Signed angle degrees {signed_angle_degrees}")
        elif angle_degrees > 20:
            turn = signed_angle_degrees * -1 / 3
            self.robot_interface.send_command("turn", turn, 15)

    def get_near_ball(self, signed_angle_degrees, angle_degrees, dist_to_ball):
        self.move_corrected(signed_angle_degrees, angle_degrees)

    def collect_ball(self, signed_angle_degrees, angle_degrees, dist_to_ball):
        self.move_corrected(signed_angle_degrees, angle_degrees)

    def start_belt(self):
        self.robot_interface.send_command("belt", 0, speedPercentage=100)

    def ejaculate(self):
        self.robot_interface.send_command("belt", 0, speedPercentage=-100)
        time.sleep(5)
        self.robot_interface.send_command("belt", 0, speedPercentage=0)

    def disconnect(self):
        print("Disconnecting from robot")
        self.robot_interface.disconnect()
        return

    def is_ready_to_ejaculate(
        self
    ):
        # Calculate the vector to the target position
        print(f"Robot pos: {self.robot_pos}, target pos: {self.dropoff_coords}\n")
        vector_to_dropoff_coords = self.dropoff_coords - self.robot_pos
        distance_to_dropoff_coords = np.linalg.norm(vector_to_dropoff_coords)


        # Calculate the signed angle between the robot's orientation and the target direction
        vector_to_goal = self.small_goal_coords - self.dropoff_coords
        signed_angle_radians = angle_between_vectors_signed(
            self.robot_vector, vector_to_goal
        )
        self.signed_angle_degrees = math.degrees(signed_angle_radians)
        self.angle_degrees = self.signed_angle_degrees
        print(f"angle to target {self.angle_degrees}")

        if (
            distance_to_dropoff_coords <= self.distance_to_delivery_point
            and self.angle_degrees < 5
        ):
            return True
        return False
