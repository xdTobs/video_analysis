import numpy as np
import sys
import math
import time
from analyse import BallNotFoundError, RobotNotFoundError, BorderNotFoundError
import RobotInterface
from utils import angle_between_vectors, angle_between_vectors_signed, coordinates_to_vector


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
        self.distance_to_delivery_point = 5  # The distance where we want to reverse belt and deliver balls
        self.collecting_balls = True

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
        self.collecting_balls = True
        if len(keypoints) == 0:
            self.collecting_balls = False  #TODO Når "True" skal robotten aflevere bolde.
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
            waypoints: np.ndarray,
            robot_pos: np.ndarray,
            robot_vector: np.ndarray,
            distance_to_closest_border: float,
            dropoff_coords: np.ndarray,
    ):
        # if we have a target and no keypoints we still want to catch last ball
        if robot_pos is None:
            raise RobotNotFoundError("No Robot to be used for program selection")
        if robot_vector is None:
            raise RobotNotFoundError("No Robot vector to be used for program selection")
        if distance_to_closest_border is None:
            raise BorderNotFoundError(
                "No distance to closest border to be used for program selection"
            )
        if not self.collecting_balls:
            self.deliver_balls_to_target(robot_pos, dropoff_coords)
        self.ball_vector = self.find_ball_vector(waypoints, robot_pos, robot_vector)
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
            #if distance_to_closest_border < self.distance_to_border_threshold:
            #   print("Close to border")
            #  self.robot_interface.send_command("stop", 0, 0)
            # #time.sleep(1)
            #self.robot_interface.send_command("move", -20, 50)
            #return

            if dist_to_ball < self.collect_ball_distance:
                self.close_to_ball = True
                print("Ball is close")
                self.collect_ball(self.signed_angle_degrees, self.angle_degrees, dist_to_ball)
                return
            else:
                print("Ball is not close")
                self.close_to_ball = False
                self.get_near_ball(self.signed_angle_degrees, self.angle_degrees, dist_to_ball)
                return
        except ConnectionError as e:
            print(f"Connection error {e}")
            return
        except Exception as e:
            print(f"Error: {e}")
            return

    def get_near_ball(self, signed_angle_degrees, angle_degrees, dist_to_ball):
        if angle_degrees < 8:
            # move 10cm at full speeed
            print(f"GET NEAR FORWARD", file=sys.stderr)
            self.robot_interface.send_command("move", 100, 50)
            print("Moving forward")
        else:
            turn = signed_angle_degrees * -1 / 3
            #  print(f"GET NEAR Turning {turn} degrees, from {signed_angle_degrees} with SPEED {speed}", file=sys.stderr)
            if angle_degrees < 20:
                self.robot_interface.send_command("turn", turn, 20)
            else:
                self.robot_interface.send_command("turn", turn, 50)

    def collect_ball(self, signed_angle_degrees, angle_degrees, dist_to_ball):
        if angle_degrees < 2:
            print(f"COLLECT FORWARD", file=sys.stderr)
            self.robot_interface.send_command("move", 30, 20)
            print("Moving forward")
        else:
            # turn 10 degrees to overcorrect so we look slightly to the side of the ball.
            turn = signed_angle_degrees * -1 / 3
            speed = 2 * (turn if turn > 0 else turn * -1)
            clamped_speed = sorted((10, speed, 40))[1]
            print(f"TURN COLLECT {str(turn)} speed {speed} clamped speed {clamped_speed}", file=sys.stderr)

            # print(f"COLLECT Turning {turn} degrees, from {signed_angle_degrees} with SPEED {speed}", file=sys.stderr)
            self.robot_interface.send_command("turn", turn, clamped_speed)
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

    def deliver_balls_to_target(self, robot_pos: np.ndarray, dropoff_cords: np.ndarray):
        if self.drive_to_point(robot_pos, dropoff_cords):
            self.stop_belt()

    def drive_to_point(self, robot_pos: np.ndarray, target_pos: np.ndarray):
        # Calculate the vector to the target position
        vector_to_position = target_pos - robot_pos
        distance_to_target = np.linalg.norm(vector_to_position)

        if distance_to_target < self.distance_threshold_min:
            print("Already at the target position or too close to move")
            return

        # Normalize the vector to get direction
        direction_to_target = vector_to_position / distance_to_target

        # Calculate the signed angle between the robot's orientation and the target direction
        signed_angle_radians = angle_between_vectors_signed(robot_pos, direction_to_target)
        signed_angle_degrees = math.degrees(signed_angle_radians)
        angle_degrees = signed_angle_degrees

        if distance_to_target <= self.distance_to_delivery_point and angle_degrees < 1:
            return True

        if distance_to_target < self.collect_ball_distance:
            self.close_to_ball = True
            print("Ball is close")
            self.collect_ball(self.signed_angle_degrees, self.angle_degrees, distance_to_target)
            return False
        else:
            print("Ball is not close")
            self.close_to_ball = False
            self.get_near_ball(self.signed_angle_degrees, self.angle_degrees, distance_to_target)
            return False
