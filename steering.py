import numpy as np
import sys
import math
import time
from analyse import BallNotFoundError, RobotNotFoundError, BorderNotFoundError
import RobotInterface
from utils import angle_between_vectors, angle_between_vectors_signed


class Steering:
    def __init__(self, online=True, host="", port=0):
        self.robot_interface = RobotInterface.RobotInterface(host, port, online)
        if online:
            self.robot_interface.connect()
        self.ball_vector = None
        self.last_target_time = 0
        self.target_position = None
        self.update_interval = 10  # Time in seconds
        self.distance_threshold_max = 500  # Distance threshold for starting the timer
        self.distance_threshold_min = 100
        self.collect_ball_distance = 250
        self.signed_angle_radians = None
        self.signed_angle_degrees = None
        self.angle_radians = None
        self.angle_degrees = None
        self.close_to_ball = False
        self.current_time = 0
        self.time_to_switch_target = 0
        self.distance_to_border_threshold = 100

    def find_ball_vector(
        self, keypoints: np.ndarray, robot_pos: np.ndarray, robot_vector: np.ndarray
    ) -> np.ndarray:
        print(
            f"Finding ball vector, robot pos: {robot_pos}, robot vector: {robot_vector}"
        )
        self.current_time = time.time()
        self.time_to_switch_target = self.current_time - self.last_target_time

        if self.target_position is not None:
            distance_to_target = np.linalg.norm(self.target_position - robot_pos)
            if self.time_to_switch_target < self.update_interval:
                if (
                    distance_to_target <= self.distance_threshold_max
                    and distance_to_target >= self.distance_threshold_min
                ):
                    print(
                        f"Using previous target position {self.target_position} as less than {self.update_interval} seconds have passed and within distance threshold."
                    )
                    return self.target_position - robot_pos
                else:
                    print(
                        f"Previous target position is too far (distance {distance_to_target}). Resetting timer."
                    )

        if len(keypoints) == 0:
            raise BallNotFoundError("No balls to be used for vector calculation")
        if robot_pos is None:
            raise RobotNotFoundError("No Robot to be used for vector calculation")

        point_distances = []

        for keypoint in keypoints:
            ball_pos = np.array(keypoint.pt)
            distance = np.linalg.norm(ball_pos - robot_pos)
            point_distances.append((keypoint, distance))

        # Sort based on distance
        point_distances = sorted(point_distances, key=lambda x: x[1])

        for idx, point_distance in enumerate(point_distances):
            if self.has_valid_path(robot_pos, robot_vector, point_distance[0].pt):
                self.target_position = point_distance[0].pt
                self.last_target_time = self.current_time
                print(
                    f"Closest valid ball is at {self.target_position}, distance {point_distance[1]}, index {idx}"
                )
                return self.target_position - robot_pos

        print("No valid path to any ball")
        self.ball_vector = point_distances[0][0].pt - robot_pos
        return point_distances[0][0].pt - robot_pos

    def has_valid_path(self, robot_pos, robot_vector, ball_pos) -> bool:
        return True

    def pick_program(
        self,
        keypoints: np.ndarray,
        robot_pos: np.ndarray,
        robot_vector: np.ndarray,
        distance_to_closest_border: float,
    ):
        # if we have a target and no keypoints we still want to catch last ball
        if len(keypoints) == 0 and not self.target_position:
            raise BallNotFoundError("No balls to be used for program selection")
        if robot_pos is None:
            raise RobotNotFoundError("No Robot to be used for program selection")
        if robot_vector is None:
            raise RobotNotFoundError("No Robot vector to be used for program selection")
        if distance_to_closest_border is None:
            raise BorderNotFoundError(
                "No distance to closest border to be used for program selection"
            )
        self.ball_vector = self.find_ball_vector(keypoints, robot_pos, robot_vector)
        if self.ball_vector is None:
            raise BallNotFoundError("No ball vector to be used for program selection")

        self.signed_angle_radians = angle_between_vectors_signed(robot_vector, self.ball_vector)  # type: ignore
        self.signed_angle_degrees = math.degrees(self.signed_angle_radians)
        self.angle_radians = angle_between_vectors(robot_vector, self.ball_vector)  # type: ignore
        self.angle_degrees = math.degrees(self.angle_radians)

        dist_to_ball = math.sqrt(self.ball_vector[0] ** 2 + self.ball_vector[1] ** 2)
        print(f"Ball vector: {self.ball_vector}")
        print(f"Ball vector length: {dist_to_ball}")

        try:
            # if distance_to_closest_border < self.distance_to_border_threshold:
            #   print("Close to border")
            #  self.robot_interface.send_command("stop", 0, 0)
            # #time.sleep(1)
            # self.robot_interface.send_command("move", -20, 50)
            # return

            if dist_to_ball < self.collect_ball_distance:
                self.close_to_ball = True
                print("Ball is close")
                self.get_near_ball(
                    self.signed_angle_degrees, self.angle_degrees, dist_to_ball
                )
                return
            else:
                print("Ball is not close")
                self.close_to_ball = False
                self.get_near_ball(
                    self.signed_angle_degrees, self.angle_degrees, dist_to_ball
                )
                return
        except ConnectionError as e:
            print(f"Connection error {e}")
            return
        except Exception as e:
            print(f"Error: {e}")
            return

    def get_near_ball(self, signed_angle_degrees, angle_degrees, dist_to_ball):
        max_speed = 100
        min_speed = 10
        max_dist = 700

        if angle_degrees < 8:
            move_speed = max(
                min_speed, min(max_speed, (dist_to_ball / max_dist) * max_speed)
            )
            self.robot_interface.send_command("move", move_speed, move_speed)
        else:
            max_turn_speed = 100
            min_turn_speed = 10
            turn_speed = max(
                min_turn_speed,
                min(max_turn_speed, (dist_to_ball / max_dist) * max_turn_speed),
            )
            turn = signed_angle_degrees * -1 / 3

            self.robot_interface.send_command("turn", turn, turn_speed)

    def start_belt(self):
        self.robot_interface.send_command("belt", 0, speedPercentage=100)

    def stop_belt(self):
        self.robot_interface.send_command("belt", 0, speedPercentage=-100)
        time.sleep(5)
        self.robot_interface.send_command("belt", 0, speedPercentage=0)

    def disconnect(self):
        print("Disconnecting from robot")
        self.robot_interface.disconnect()

    def deliver_balls_to_target(self, target_goal: np.ndarray):
        # Calculate the direction vector from the current position to the target position
        direction_vector = target_goal - self.robot_pos
        # Normalize the direction vector
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        # Calculate the distance to the target position
        distance = np.linalg.norm(target_goal - self.robot_pos)
        # Send a command to the robot to move in the direction of the target position
        self.robot_interface.send_command("move", direction_vector, distance)
