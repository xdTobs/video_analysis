import numpy as np
import math
import time
from analyse import BallNotFoundError, RobotNotFoundError
import RobotInterface
from utils import angle_between_vectors, angle_between_vectors_signed

class Steering():
    def __init__(self, online = True, host = "", port = 0):
        self.robot_interface = RobotInterface.RobotInterface(host, port, online)
        if online:
            self.robot_interface.connect()
        self.ball_vector = None
        self.last_target_time = 0
        self.target_position = None
        self.update_interval = 5  # Time in seconds
        self.distance_threshold = 100  # Distance threshold for starting the timer
    def find_ball_vector(self, keypoints: np.ndarray, robot_pos: np.ndarray, robot_vector: np.ndarray) -> np.ndarray:
        print(f"Finding ball vector, robot pos: {robot_pos}, robot vector: {robot_vector}")
        current_time = time.time()

        if self.target_position is not None:
            distance_to_target = np.linalg.norm(self.target_position - robot_pos)
            if current_time - self.last_target_time < self.update_interval:
                if distance_to_target <= self.distance_threshold:
                    print(f"Using previous target position {self.target_position} as less than {self.update_interval} seconds have passed and within distance threshold.")
                    return self.target_position - robot_pos
                else:
                    print(f"Previous target position is too far (distance {distance_to_target}). Resetting timer.")

        
        
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
                self.last_target_time = current_time
                print(f"Closest valid ball is at {self.target_position}, distance {point_distance[1]}, index {idx}")
                return self.target_position - robot_pos
        
        print("No valid path to any ball")
        self.ball_vector = point_distances[0][0].pt - robot_pos
        return point_distances[0][0].pt - robot_pos
    
    def has_valid_path(self, robot_pos, robot_vector, ball_pos) -> bool:
        return True
    
    def pick_program(self, keypoints: np.ndarray, robot_pos: np.ndarray, robot_vector: np.ndarray):
        if len(keypoints) == 0:
            raise BallNotFoundError("No balls to be used for program selection")
        if robot_pos is None:
            raise RobotNotFoundError("No Robot to be used for program selection")
        if robot_vector is None:
            raise RobotNotFoundError("No Robot vector to be used for program selection")
        
            
        self.ball_vector = self.find_ball_vector(keypoints, robot_pos, robot_vector)
        if self.ball_vector is None:
            raise BallNotFoundError("No ball vector to be used for program selection")
        
        signed_angle_radians = angle_between_vectors_signed(robot_vector, self.ball_vector)  # type: ignore
        signed_angle_degrees = math.degrees(signed_angle_radians)
        angle_radians = angle_between_vectors(robot_vector, self.ball_vector)  # type: ignore
        angle_degrees = math.degrees(angle_radians)
        
        dist_to_ball = math.sqrt(self.ball_vector[0] ** 2 + self.ball_vector[1] ** 2)
        print(f"Ball vector: {self.ball_vector}")
        print(f"Ball vector length: {dist_to_ball}")
        
        try:
            if dist_to_ball < 250:
                print("Ball is close")
                self.collect_ball(signed_angle_degrees, angle_degrees)
                return
            else:
                print("Ball is not close")
                self.get_near_ball(signed_angle_degrees, angle_degrees)
                return
        except ConnectionError as e:
            print(f"Connection error {e}")
            return
        except Exception as e:
            print(f"Error: {e}")
            return
        return
    
    def get_near_ball(self, signed_angle_degrees, angle_degrees):
        if angle_degrees < 10:
            #move 10cm at full speeed
            self.robot_interface.send_command("move", 100, 100)
            print("Moving forward")
        else:
            print(f"Turning {signed_angle_degrees} degrees")
            self.robot_interface.send_command("turn", signed_angle_degrees * -1 / 3, 15)
        
        pass
    
    def collect_ball(self,signed_angle_degrees, angle_degrees):
        if angle_degrees < 10:
            self.robot_interface.send_command("move", 30, 30)
            print("Moving forward")
        else:
            print(f"Turning {signed_angle_degrees} degrees")
            self.robot_interface.send_command("turn", signed_angle_degrees * -1 / 3, 15)
        pass
    
    def start_belt(self):
        self.robot_interface.send_command("belt", 0, speedPercentage=100)
        
    def stop_belt(self):
        self.robot_interface.send_command("belt", 0, speedPercentage=-100)
        time.sleep(5)
        self.robot_interface.send_command("belt", 0, speedPercentage=0)
    
    
