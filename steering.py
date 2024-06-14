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
        self.update_interval = 1  # Time in seconds
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
            if distance_to_closest_border < 30:
                print("Close to border")
                self.robot_interface.send_command("move", 0, 0)
                time.sleep(1)
                self.robot_interface.send_command("move", -20, 0)
                return

            if dist_to_ball < self.collect_ball_distance:
                self.close_to_ball = True
                print("Ball is close")
                self.collect_ball(self.signed_angle_degrees, self.angle_degrees)
                return
            else:
                print("Ball is not close")
                self.close_to_ball = False
                self.get_near_ball(self.signed_angle_degrees, self.angle_degrees)
                return
        except ConnectionError as e:
            print(f"Connection error {e}")
            return
        except Exception as e:
            print(f"Error: {e}")
            return

    def get_near_ball(self, signed_angle_degrees, angle_degrees):
        if angle_degrees < 10:
            # move 10cm at full speeed
            print(f"GET NEAR FORWARD", file=sys.stderr)
            self.robot_interface.send_command("move", 100, 100)
            print("Moving forward")
        else:
            turn = signed_angle_degrees * -1 / 3
          #  print(f"GET NEAR Turning {turn} degrees, from {signed_angle_degrees} with SPEED {speed}", file=sys.stderr)
            self.robot_interface.send_command("turn", turn, 50)

    def collect_ball(self, signed_angle_degrees, angle_degrees):
        if angle_degrees < 5:
            print(f"COLLECT FORWARD", file=sys.stderr)
            self.robot_interface.send_command("move", 30, 30)
            print("Moving forward")
        else:
            # turn 10 degrees to overcorrect so we look slightly to the side of the ball.
            turn = signed_angle_degrees * -1 / 3 
    
         #   print(f"COLLECT Turning {turn} degrees, from {signed_angle_degrees} with SPEED {speed}", file=sys.stderr)
            self.robot_interface.send_command("turn",turn , 20)
        pass

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
    
    def deliver_balls_to_target(self, target_goal: np.ndarray):
        # Calculate the direction vector from the current position to the target position
        direction_vector = target_goal - self.robot_pos
        # Normalize the direction vector
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        # Calculate the distance to the target position
        distance = np.linalg.norm(target_goal - self.robot_pos)
        # Send a command to the robot to move in the direction of the target position
        self.robot_interface.send_command("move", direction_vector, distance)