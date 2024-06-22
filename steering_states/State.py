from video_analysis.analyse import Analyse
import time
import math
import numpy as np
from utils import angle_between_vectors

class State():
    def __init__(self, analyser : Analyse):
        self.start_time = time.time()
        self.analyser = analyser
        
        
    def on_frame(self):
        pass
    def swap_state(self):
        pass
    
class PathingState(State):
    def __init__(self, analyser : Analyse, path : list):
        super().__init__(analyser)
        self.path = path
        self.timeout = 30 # Seconds
        self.steering_vector = path[0]
        
    def on_frame(self):
        
        pass
    
    def swap_state(self):
        # Check timeout
        
        if time.time() - self.start_time > self.timeout:
            #TODO Move / create
            if len(self.analyser.keypoints) == 0:
                return DeliveringState(self.analyser, self.analyser.create_path())
            return PathingState(self.analyser, self.analyser.create_path())
        
        if len(self.path) == 1:
            #TODO Check that this passes absolute coords and not relative
            if len(self.analyser.keypoints) == 0:
                return DeliveringState(self.analyser, self.analyser.create_path())
            return CollectionState(self.analyser, self.path[0])
        
        if math.degrees(angle_between_vectors(self.analyser.robot_vector,self.steering_vector)) > 100:
            #TODO Might need to use a different arugment for the path, might need to be absolute
            return ReversingState(self.analyser, self.path)
        
        return self
        
class ReversingState(State):
    def __init__(self, analyser: Analyse, path: list):
        super().__init__(analyser)
        self.path = path
        self.timeout = 30 # seconds
        
        if len(path) > 1:
            #TODO Check if this is the correct vector
            self.result_vector = path[1] - path[0]
        else:
            #TODO This might give errors, but we hope that just turning some way and then swapping to collection state is fine, might be a problem if we want to reverse befroe delivering
            self.result_vector = np.array([1,0])
        
    def on_frame(self):
        pass
    
    def swap_state(self):
        #Check timeout
        if self.timeout < self.analyser.get_time() - self.start_time:
            #TODO Move / create
            if len(self.analyser.keypoints) == 0:
                return DeliveringState(self.analyser, self.analyser.create_path())
            return PathingState(self.analyser, self.analyser.create_path())
        
        if math.degrees(angle_between_vectors(self.analyser.robot_vector,self.result_vector)) < 5:
            #TODO Might need to use a different arugment for the path, might need to be absolute
            #TODO Might also need to skip pathing state and go straight to collection state
            if len(self.analyser.keypoints) == 0:
                return DeliveringState(self.analyser, self.analyser.create_path())
            return PathingState(self.analyser, self.path)

        return self
        


class CollectionState(State):
    def __init__(self,analyser: Analyse, ball_point: list):
        super().__init__(analyser)
        self.path = ball_point
        self.distance_before_swap = 60 # px
        self.timeout = 30 # seconds
        self.speed = 0 # % of max speed
        
    def on_frame(self):
        self.analyser.get_speed(np.norm(self.ball_point - self.analyser.robot_pos))
        
        pass
    def swap_state(self):
        #Check timeout
        if self.timeout < self.analyser.get_time() - self.start_time:
            #TODO Move / create
            if len(self.analyser.keypoints) == 0:
                return DeliveringState(self.analyser, self.analyser.create_path())
            return PathingState(self.analyser, self.analyser.create_path())
        
        #TODO Check if we need to check for a certain distance to the ball
        if np.linalg.norm(self.analyser.robot_pos - self.path) < self.distance_before_swap:
            #TODO Might need to use a different arugment for the path, might need to be absolute
            if len(self.analyser.keypoints) == 0:
                return DeliveringState(self.analyser, self.analyser.create_path())
            return PathingState(self.analyser, self.analyser.create_path())
        
        return self

class DeliveringState(State):
    def __init__(self, analyser: Analyse, path: list):
        super().__init__(analyser)
        self.path = path
        
        
    def on_frame(self):
        pass
    def swap_state(self):
        if len(self.analyser.keypoints) != 0:
            return PathingState(self.analyser, self.analyser.create_path())
        return self