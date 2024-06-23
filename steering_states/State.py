import utils
from analyse import Analyse
import time
import math
import numpy as np
from utils import angle_between_vectors, angle_between_vectors_signed
from steering_states.SteeringUtils import SteeringUtils


class State():
    def __init__(self, analyser : Analyse, steering_utils : SteeringUtils):
        self.steering = steering_utils
        self.start_time = time.time()
        self.analyser = analyser


    def on_frame(self):
        pass
    def swap_state(self):
        pass

class PathingState(State):
    def __init__(self, analyser : Analyse, path : list, steering : SteeringUtils):
        super().__init__(analyser, steering)
        self.path = path
        self.timeout = 30 # Seconds
        self.steering_vector = path[0]

    def on_frame(self):
        if self.analyser.are_coordinates_close(self.path[0]) and len(self.path) > 1:
            self.path.pop(0)
        elif self.analyser.can_target_ball_directly(self.analyser.robot_pos, self.path[-1]):
            while len(self.path) > 1:
                self.path.pop(0)
        self.steering_vector = self.path[0]
        signed_angle_degree = math.degrees(angle_between_vectors(self.analyser.robot_vector, self.steering_vector))
        self.steering.move_corrected(signed_angle_degree, 30)
        pass

    def swap_state(self):
        # Check timeout

        if time.time() - self.start_time > self.timeout:
            #TODO Move / create
            if len(self.analyser.keypoints) == 0:
                return DeliveringState(self.analyser, self.analyser.create_path(), self.steering)
            return PathingState(self.analyser, self.analyser.create_path(), self.steering)

        if len(self.path) == 1:
            #TODO Check that this passes absolute coords and not relative
            if len(self.analyser.keypoints) == 0:
                return DeliveringState(self.analyser, self.analyser.create_path(), self.steering)

            return CollectionState(self.analyser, [self.path[0]], self.steering)

        if math.degrees(angle_between_vectors(self.analyser.robot_vector,self.steering_vector)) > 100:
            #TODO Might need to use a different arugment for the path, might need to be absolute
            return ReversingState(self.analyser, self.path, self.steering)

        return self

class ReversingState(State):
    def __init__(self, analyser: Analyse, path: list, steering: SteeringUtils):
        super().__init__(analyser,steering)
        self.path = path
        self.timeout = 30 # seconds

        if len(path) > 1:
            #TODO Check if this is the correct vector
            self.result_vector = path[1] - path[0]
        else:
            #TODO This might give errors, but we hope that just turning some way and then swapping to collection state is fine, might be a problem if we want to reverse befroe delivering
            self.result_vector = np.array([1,0])

    def on_frame(self):
        steering_vector = self.path[0]

        if self.analyser.are_coordinates_close(self.path[0]):
            self.steering.turn(10, 10)
        else:
            signed_angle_degree = math.degrees(angle_between_vectors_signed(self.analyser.robot_vector, steering_vector))
            if signed_angle_degree < 0:
                signed_angle_degree -= 180
            else:
                signed_angle_degree += 180
            self.steering.move_corrected(signed_angle_degree, -30)
        pass

    def swap_state(self):
        #Check timeout
        if self.timeout < time.time() - self.start_time:
            #TODO Move / create
            if len(self.analyser.keypoints) == 0:
                return DeliveringState(self.analyser, self.analyser.create_path(), self.steering)
            return PathingState(self.analyser, self.analyser.create_path(), self.steering)

        if self.analyser.are_coordinates_close(self.path[0]) and math.degrees(angle_between_vectors(self.analyser.robot_vector,self.result_vector)) < 30:
            #TODO Might need to use a different arugment for the path, might need to be absolute
            #TODO Might also need to skip pathing state and go straight to collection state
            if len(self.analyser.keypoints) == 0:
                return DeliveringState(self.analyser, self.analyser.create_path(), self.steering)
            return PathingState(self.analyser, self.path, self.steering)

        return self



class CollectionState(State):
    def __init__(self,analyser: Analyse, ball_point: list, steering: SteeringUtils):
        super().__init__(analyser,steering)
        self.path = ball_point
        self.distance_before_swap = 60 # px
        self.timeout = 30 # seconds
        self.speed = 0 # % of max speed
        self.safe_distance_middle = 100 # px

    def on_frame(self):
        ball_point = self.path[0]
        self.speed = self.analyser.get_speed(np.linalg.norm(ball_point - self.analyser.robot_pos))
        print(f"Ball point: {ball_point}, robot vector: {self.analyser.robot_vector}")
        #TODO Assuming relative coords, might be getting absolute
        signed_angle_degree = math.degrees(angle_between_vectors_signed(self.analyser.robot_vector, ball_point))
        self.steering.move_corrected(signed_angle_degree, self.speed)

        if self.analyser.distance_to_middle < self.safe_distance_middle:
            self.steering.start_belt()
        else:
            self.steering.stop_belt()
        pass

    def swap_state(self):
        #Check timeout
        if self.timeout < time.time() - self.start_time:
            #TODO Move / create
            if len(self.analyser.keypoints) == 0:
                return DeliveringState(self.analyser, self.analyser.create_path(), self.steering)
            return PathingState(self.analyser, self.analyser.create_path(), self.steering)

        #TODO Check if we need to check for a certain distance to the ball
        if np.linalg.norm(self.analyser.robot_pos - self.path) < self.distance_before_swap:
            #TODO Might need to use a different arugment for the path, might need to be absolute
            if len(self.analyser.keypoints) == 0:
                return DeliveringState(self.analyser, self.analyser.create_path(), self.steering)
            return PathingState(self.analyser, self.analyser.create_path(), self.steering)

        return self

class DeliveringState(State):
    def __init__(self, analyser: Analyse, path: list, steering: SteeringUtils):
        super().__init__(analyser, steering)
        self.path = path

    def on_frame(self):
        signed_angle_degree = math.degrees(angle_between_vectors_signed(self.analyser.robot_vector, self.analyser.dropoff_coords))
        self.steering.move_corrected(signed_angle_degree, 15)
        self.steering.stop_belt()
        pass
    def swap_state(self):
        if self.analyser.are_coordinates_close(self.path[0]):
            return ReleaseBallsState(self.analyser, self.steering)
        return self

class ReleaseBallsState(State):
    def __init__(self, analyser: Analyse, steering: SteeringUtils):
        super().__init__(analyser, steering)
        self.timeout = 10

    def on_frame(self):
        if utils.is_coordinates_close(self.analyser.robot_pos, self.analyser.dropoff_coords, 100):
            self.steering.reverse_belt()
        pass
    def swap_state(self):
        if self.timeout < time.time() - self.start_time:
            return PathingState(self.analyser, self.analyser.create_path(), self.steering)
        return self
