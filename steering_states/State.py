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
            return PathingState(self.analyser, self.analyser.create_path())
        
        if len(self.path) == 1:
            #TODO Check that this passes absolute coords and not relative
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
            return self.analyser.create_path()
        


class CollectionState(State):
    def __init__(self,analyser: Analyse, ball_point: list):
        super().__init__(analyser)
        self.path = ball_point

        self.timeout = 30 # seconds
        
    def on_frame(self):
        pass

class DeliveringState(State):
    def __init__(self, analyser: Analyse, path: list):
        super().__init__(analyser)
        self.path = path
        
        
    def on_frame(self):
        pass