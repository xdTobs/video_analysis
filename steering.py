import numpy as np
import sys
import math
import time
from analyse import RobotNotFoundError, BorderNotFoundError
import RobotInterface
from utils import angle_between_vectors, angle_between_vectors_signed
from analyse import Analyse


class Steering:
    def __init__(self, online=True, host="", port=0):
        self.robot_interface = RobotInterface.RobotInterface(host, port, online)
        if online:
            self.robot_interface.connect()
        self.ball_vector = None
        self.is_ball_close_to_border = False
        self.last_target_time = 0
        self.target_position = None
        self.update_interval = 30  # Time in seconds
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
        self.is_collecting_balls = True
        # first is if we are turning second is if we are turning right
        self.turn_start = None

    # checks if we can go to ball without crashing into the mid cross
    def check_no_obstacles(
        self, robot_pos: np.ndarray, target_pos: np.ndarray, obstacles: np.ndarray
    ) -> bool:
        # if robot_pos is None or target_pos is None or obstacles is None:
        #     return False

        # # Define the vector from the robot to the target
        # direction_vector = target_pos - robot_pos
        # distance_to_target = np.linalg.norm(direction_vector)
        # direction_unit_vector = direction_vector / distance_to_target

        # # Check each obstacle
        # for obstacle in obstacles:
        #     obstacle_vector = obstacle - robot_pos
        #     obstacle_distance = np.linalg.norm(obstacle_vector)
        #     obstacle_unit_vector = obstacle_vector / obstacle_distance

        #     # Project the obstacle vector onto the direction vector
        #     projection_length = np.dot(obstacle_vector, direction_unit_vector)
        #     if 0 < projection_length < distance_to_target:
        #         # Find the perpendicular distance from the obstacle to the direction line
        #         perpendicular_distance = np.linalg.norm(
        #             obstacle_vector - projection_length * direction_unit_vector
        #         )
        #         if perpendicular_distance < self.distance_to_border_threshold:
        #             return False  # Obstacle detected within the path
        return True  # No obstacles detected

    def find_ball_vector(
        self,
        keypoints: np.ndarray,
        robot_pos: np.ndarray,
        robot_vector: np.ndarray,
        border_mask,
        safepoint_list: np.ndarray
    ) -> np.ndarray:
        print("safepoints", safepoint_list)
        self.current_time = time.time()
        self.time_to_switch_target = self.current_time - self.last_target_time

        if self.target_position is not None:
            distance_to_target = np.linalg.norm(self.target_position - robot_pos)
            if self.time_to_switch_target < self.update_interval:
                if (
                    self.distance_threshold_max
                    >= distance_to_target
                    >= self.distance_threshold_min
                ):
                    return self.target_position - robot_pos
                else:
                    print(
                        f"Previous target position is too far (distance {distance_to_target}). Resetting timer."
                    )
        # self.is_collecting_balls = True
        if len(keypoints) == 0:
            return None

            # self.is_collecting_balls = False  #TODO Når "True" skal robotten aflevere bolde.
            # return
        if robot_pos is None:
            raise RobotNotFoundError("No Robot to be used for vector calculation")

        point_distances = []

        for keypoint in keypoints:
            ball_pos = np.array(keypoint.pt)
            distance = np.linalg.norm(ball_pos - robot_pos)
            point_distances.append((keypoint, distance))

        # Sort based on distance
        point_distances = sorted(point_distances, key=lambda x: x[1])
        is_not_blocked = self.check_no_obstacles(
            robot_pos=robot_pos,
            target_pos=ball_pos,
            obstacles=Analyse.find_cross_bounding_rectangle(border_mask),
        )
        print("is blocked: ", is_not_blocked)

        for idx, point_distance in enumerate(point_distances):
            if self.has_valid_path(robot_pos, robot_vector, point_distance[0].pt):
                self.target_position = point_distance[0].pt
                self.last_target_time = self.current_time
                self.closest_safe_point_to_ball = self.find_closest_safe_point_to_ball(self.target_position, safepoint_list)
                print(
                    f"Closest valid ball is at {self.target_position}, distance {point_distance[1]}, index {idx}"
                )
                print("Closest safe point: ", self.closest_safe_point_to_ball, "\n")

                return self.target_position - robot_pos

        print("No valid path to any ball")
        self.ball_vector = point_distances[0][0].pt - robot_pos
        return point_distances[0][0].pt - robot_pos

    def has_valid_path(self, robot_pos, robot_vector, ball_pos) -> bool:
        return True
    
    def find_closest_safe_point_to_ball(self, ball_pos: np.ndarray, safepoint_list: np.ndarray) -> np.ndarray:
        closest_distance = sys.maxsize
        closest_point = None
        for point in safepoint_list:
            distance = np.linalg.norm(ball_pos - point)
            if distance < closest_distance:
                closest_distance = distance
                closest_point = point
        return closest_point

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

    def pick_program(
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

        self.ball_vector = self.find_ball_vector(
            keypoints, robot_pos, robot_vector, border_mask, safepoint_list
        )
        if not self.is_collecting_balls:
            self.deliver_balls_to_target(robot_vector, dropoff_coords, robot_pos)
        if self.ball_vector is None:
            self.is_collecting_balls = False
        else:
            self.is_collecting_balls = True

        if self.is_collecting_balls:
            self.signed_angle_radians = angle_between_vectors_signed(robot_vector, self.ball_vector)  # type: ignore
            self.signed_angle_degrees = math.degrees(self.signed_angle_radians)
            self.angle_radians = angle_between_vectors(robot_vector, self.ball_vector)  # type: ignore
            self.angle_degrees = math.degrees(self.angle_radians)

        dist_to_ball = math.sqrt(self.ball_vector[0] ** 2 + self.ball_vector[1] ** 2)

        self.is_ball_close_to_border = self.calculate_is_ball_close_to_borders(
            self.target_position, corners
        )
        print(f"Ball vector: {self.ball_vector}")
        print(f"Ball vector length: {dist_to_ball}")
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
                    self.signed_angle_degrees, self.angle_degrees, dist_to_ball
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
        if angle_degrees < 2:
            self.robot_interface.send_command("move", 100, speed)
        elif 2 <= angle_degrees <= 8:
            self.robot_interface.send_command("move-corrected", -1 * signed_angle_degrees, speed)
            print(f"Signed angle degrees {signed_angle_degrees}")
        elif angle_degrees > 8:
            turn = signed_angle_degrees * -1 / 3
            self.robot_interface.send_command("turn", turn, 30)

    def get_near_ball(self, signed_angle_degrees, angle_degrees, dist_to_ball):
        self.move_corrected(signed_angle_degrees, angle_degrees, 100)

    def collect_ball(self, signed_angle_degrees, angle_degrees, dist_to_ball):
        self.move_corrected(signed_angle_degrees, angle_degrees, 100)

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
