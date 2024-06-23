import numpy as np
import sys
import math
import time
from analyse import RobotNotFoundError, BorderNotFoundError
import RobotInterface
from utils import angle_between_vectors, angle_between_vectors_signed
from analyse import Analyse
from collections import deque
from enum import Enum
from steering_states.SteeringUtils import SteeringUtils


from steering_states.State import (
    State,
    PathingState,
    ReversingState,
    CollectionState,
    DeliveringState,
)


class stateEnum(Enum):
    NONE_STATE = -1
    PATHING_STATE = 0
    COLLECTING_STATE = 1
    DELIVERING_STATE = 2
    REVERSING_STATE = 3


class Steering:
    def __init__(self, analyser: Analyse, online=True, host="", port=0):

        self.robot_interface = RobotInterface.RobotInterface(host, port, online)
        if online:
            self.robot_interface.connect()
        self.analyser = analyser
        self.state: State = None
        self.state_enum = None
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
        self.close_to_ball = False
        self.current_time = 0
        self.time_to_switch_target = 0
        self.distance_to_border_threshold = 100
        self.distance_to_delivery_point = (
            30  # The distance where we want to reverse belt and deliver balls
        )
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

    def on_frame(self):

        if self.state is None and self.analyser.safepoint_list is not None:
            path = self.analyser.create_path()
            self.state: State = PathingState(
                self.analyser, path, SteeringUtils(self.robot_interface)
            )
            self.state_enum = stateEnum.PATHING_STATE
        if self.state is None:
            return

        self.state.on_frame()
        self.state = self.state.swap_state()
        return

    # checks if we can go to ball without crashing into the mid cross
    def check_no_obstacles(
        self, robot_pos: np.ndarray, target_pos: np.ndarray, obstacles: np.ndarray
    ) -> bool:
        return True

    def stop_belt(self):
        self.robot_interface.send_command("belt", 0, speedPercentage=-100)
        time.sleep(1)
        self.robot_interface.send_command("belt", 0, speedPercentage=0)

    def stop_motors(self):
        time.sleep(1)
        self.robot_interface.send_command("move", 0, speedPercentage=0)
        time.sleep(1)
        self.robot_interface.send_command("turn", 0, speedPercentage=0)
        time.sleep(1)
        self.robot_interface.send_command("belt", 0, speedPercentage=0)

    def disconnect(self):
        self.robot_interface.disconnect()
        return

    def follow_path(
        self, keypoints: np.ndarray, robot_pos: np.ndarray, safepoint_list: np.ndarray
    ) -> np.ndarray:
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

    def should_switch_target(self, robot_pos: np.ndarray, ball_pos: np.ndarray) -> bool:
        if ball_pos is None:
            return True
        distance_to_ball = np.linalg.norm(ball_pos - robot_pos)
        if (
            self.is_target_expired()
            or self.target_ball is None
            or distance_to_ball < self.distance_threshold_min
        ):
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

    def pick_program_pipeline(
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

        dist_to_target = math.sqrt(
            self.steering_vector[0] ** 2 + self.steering_vector[1] ** 2
        )
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
