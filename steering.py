import numpy as np
from analyse import BallNotFoundError, RobotNotFoundError

class Steering():
    def __init__(self):
        x = 1
        
    def find_ball_vector(self, keypoints : np.ndarray, robot_pos : np.ndarray, robot_vector : np.ndarray) -> np.ndarray:
        if len(keypoints) == 0:
            raise BallNotFoundError("No balls to be used for vector calculation")
        if robot_pos is None:
            raise RobotNotFoundError("No Robot to be used for vector calculation")
        
        point_distances = []
        for keypoint in keypoints:
            ball_pos = np.array(keypoint.pt)
            distance = np.linalg.norm(ball_pos - robot_pos)
            point_distances.append((keypoint, distance))
            
            #Sort based on distance
            point_distances = sorted(point_distances, key=lambda x: x[1])
            
            for idx, point_distance in enumerate(point_distances):
                if self.has_valid_path(robot_pos, robot_vector, point_distance[0].pt):
                    print(f"Closest valid ball is at {point_distance[0].pt}, distance {point_distance[1]}, index {idx}")
                    return point_distance[0].pt - robot_pos
                    break
            print("No valid path to any ball")
            

        return point_distances[0] - robot_pos
    
    def has_valid_path(self, robot_pos, robot_vector, ball_pos) -> bool:
        return True
        