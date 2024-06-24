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
)


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

        path = self.analyser.create_path()
        if path is not None and self.state is None and self.analyser.safepoint_list is not None:
            self.state: State = PathingState(
                self.analyser, path, SteeringUtils(self.robot_interface)
            )
        if path is None:
            return
        if self.state is None and self.analyser.safepoint_list is None:
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
