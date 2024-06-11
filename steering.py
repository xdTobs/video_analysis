import numpy as np
import time
from analyse import BallNotFoundError, RobotNotFoundError

class Steering():
    def __init__(self):
        self.last_target_time = 0
        self.target_position = None
        self.update_interval = 5  # Time in seconds
        self.distance_threshold = 50  # Distance threshold for starting the timer

    def find_ball_vector(self, keypoints: np.ndarray, robot_pos: np.ndarray, robot_vector: np.ndarray) -> np.ndarray:
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
                if point_distance[1] <= self.distance_threshold:
                    self.target_position = point_distance[0].pt
                    self.last_target_time = current_time
                    print(f"Closest valid ball is at {self.target_position}, distance {point_distance[1]}, index {idx}")
                    return self.target_position - robot_pos
                else:
                    print(f"Valid ball found but is too far (distance {point_distance[1]}), not updating target.")
        
        print("No valid path to any ball")
        return point_distances[0][0].pt - robot_pos
    
    def has_valid_path(self, robot_pos, robot_vector, ball_pos) -> bool:
        return True
