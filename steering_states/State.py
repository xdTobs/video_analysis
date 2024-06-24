from analyse import Analyse
import time
import math
import numpy as np
from utils import (
    angle_between_vectors,
    angle_between_vectors_signed,
)
from steering_states.SteeringUtils import SteeringUtils


class State:
    def __init__(self, analyser: Analyse, steering_utils: SteeringUtils):
        self.steering = steering_utils
        self.start_time = time.time()
        self.analyser = analyser
        self.path = None

    def on_frame(self):
        self.steering.start_belt()
        pass

    def swap_state(self):
        pass


class PathingState(State):
    def __init__(self, analyser: Analyse, path: list, steering: SteeringUtils):
        super().__init__(analyser, steering)
        self.path = path
        self.timeout = 30  # Seconds
        self.steering_vector = path[0]

    def on_frame(self):

        # self.steering.stop_belt()
        if self.analyser.is_point_close(self.path[0]) and len(self.path) > 1:
            self.path.pop(0)
        # elif self.analyser.can_target_ball_directly(self.analyser.robot_pos, self.path[-1]):
        #     while len(self.path) > 1:
        #         self.path.pop(0)
        self.steering_vector = self.path[0] - self.analyser.robot_pos
        signed_angle_degree = math.degrees(
            angle_between_vectors_signed(
                self.analyser.robot_vector, self.steering_vector
            )
        )

        self.steering.move_corrected(signed_angle_degree, 30)
        pass

    def swap_state(self):
        # Check timeout

        if time.time() - self.start_time > self.timeout:
            return PathingState(
                self.analyser, self.analyser.create_path(), self.steering
            )

        if len(self.path) == 1:
            # TODO Check that this passes absolute coords and not relative
            if len(self.analyser.keypoints) == 0:
                return SafePointDeliveryState(
                    self.analyser, self.analyser.create_path(), self.steering
                )

            return CollectionState(self.analyser, [self.path[0]], self.steering)

        border_distance, _ = self.analyser.calculate_distance_to_closest_border(
            self.analyser.robot_pos
        )
        if (
            math.degrees(
                angle_between_vectors(self.analyser.robot_vector, self.steering_vector)
            )
            > 100
            and border_distance < 100
        ):
            # TODO Might need to use a different arugment for the path, might need to be absolute
            return ReversingState(self.analyser, self.path, self.steering)

        return self


class ReversingState(State):
    def __init__(self, analyser: Analyse, path: list, steering: SteeringUtils):
        super().__init__(analyser, steering)
        self.path = path
        self.timeout = 30  # seconds

        if len(path) > 1:
            # TODO Check if this is the correct vector
            self.result_vector = path[1] - path[0]
        else:
            # TODO This might give errors, but we hope that just turning some way and then swapping to collection state is fine, might be a problem if we want to reverse befroe delivering
            self.result_vector = np.array([1, 0])

    def on_frame(self):
        # self.steering.stop_belt()
        steering_vector = self.path[0] - self.analyser.robot_pos

        if self.analyser.is_point_close(self.path[0]):
            signed_angle_degree = math.degrees(
                angle_between_vectors_signed(
                    self.analyser.robot_vector, self.result_vector
                )
            )
            self.steering.turn(-1 * signed_angle_degree, 10)
        else:
            signed_angle_degree = math.degrees(
                angle_between_vectors_signed(
                    self.analyser.robot_vector, steering_vector
                )
            )
            if signed_angle_degree < 0:
                signed_angle_degree += 180
            else:
                signed_angle_degree -= 180
            print(f"Reverse angle: {signed_angle_degree}")
            self.steering.move_corrected(signed_angle_degree, -30)
        pass

    def swap_state(self):
        # Check timeout
        # if self.timeout < time.time() - self.start_time:
        #     #TODO Move / create
        #     if len(self.analyser.keypoints) == 0:
        #         return DeliveringState(self.analyser, self.analyser.create_path(), self.steering)
        #     return PathingState(self.analyser, self.analyser.create_path(), self.steering)
        # print("LOG SWAPPING", math.degrees(angle_between_vectors(self.analyser.robot_vector,self.result_vector)))
        # if math.degrees(angle_between_vectors(self.analyser.robot_vector,self.result_vector)) < 90:
        #     #TODO Might need to use a different arugment for the path, might need to be absolute
        #     #TODO Might also need to skip pathing state and go straight to collection state
        #     print("LOG LEN", len(self.analyser.keypoints), self.analyser.keypoints)

        #     if len(self.analyser.keypoints) == 0:
        #         return DeliveringState(self.analyser, self.analyser.create_path(), self.steering)
        #     return PathingState(self.analyser, self.path, self.steering)
        if (
            math.degrees(
                angle_between_vectors(self.analyser.robot_vector, self.result_vector)
            )
            < 45
        ):
            return PathingState(self.analyser, self.path, self.steering)
        return self


class SafePointDeliveryState(State):
    def __init__(self, analyser: Analyse, path: list, steering: SteeringUtils):
        super().__init__(analyser, steering)
        self.path = []
        adjustment = (
            -30 if self.analyser.robot_pos[1] < self.analyser.dropoff_coords[1] else 30
        )
        print("IS NORTH" if adjustment > 0 else "IS SOUTH")
        self.closest_safepoint = self.analyser.safepoint_list[9] - np.array(
            [40, adjustment]
        )
        self.is_close_to_safepoint = False
        self.goal_vector_degrees = None

    def on_frame(self):
        self.steering_vector = self.closest_safepoint - self.analyser.robot_pos
        signed_angle_degree = math.degrees(
            angle_between_vectors_signed(
                self.analyser.robot_vector, self.steering_vector
            )
        )
        self.path = [self.closest_safepoint, self.analyser.dropoff_coords]

        if self.analyser.are_coordinates_close(self.steering_vector, 5):
            self.is_close_to_safepoint = True

        self.goal_vector_degrees = math.degrees(
            angle_between_vectors_signed(
                self.analyser.robot_vector,
                self.analyser.dropoff_coords - self.analyser.robot_pos,
            )
        )
        if self.is_close_to_safepoint:
            print("CLOSE TO SAFEP")
            if abs(self.goal_vector_degrees) > 1:
                print("CLOSE AND TURNING")
                self.steering.turn(-1 * self.goal_vector_degrees, 3)
            print(self.goal_vector_degrees, self.analyser.robot_vector)
        else:
            self.steering.move_corrected(signed_angle_degree, 5, turn_speed=15)

    def swap_state(self):
        if self.is_close_to_safepoint and abs(self.goal_vector_degrees) < 2:
            return DeliveryPointDeliveringState(self.analyser, self.steering)
        return self


class CollectionState(State):
    def __init__(self, analyser: Analyse, ball_point: list, steering: SteeringUtils):
        super().__init__(analyser, steering)
        self.path = ball_point
        self.distance_before_swap = 60  # px
        self.timeout = 30  # seconds
        self.speed = 0  # % of max speed
        self.safe_distance_middle = 100  # px

    def on_frame(self):
        ball_point = self.path[0]
        self.speed = self.analyser.get_speed(
            np.linalg.norm(ball_point - self.analyser.robot_pos)
        )
        print(f"Ball point: {ball_point}, robot vector: {self.analyser.robot_vector}")
        # TODO Assuming relative coords, might be getting absolute

        self.steering_vector = ball_point - self.analyser.robot_pos

        signed_angle_degree = math.degrees(
            angle_between_vectors_signed(
                self.analyser.robot_vector, self.steering_vector
            )
        )
        self.steering.move_corrected(signed_angle_degree, self.speed)
        self.steering.start_belt()

    def swap_state(self):
        # Check timeout
        if self.timeout < time.time() - self.start_time:
            # TODO Move / create
            return PathingState(
                self.analyser, self.analyser.create_path(), self.steering
            )

        # TODO Check if we need to check for a certain distance to the ball
        if np.linalg.norm(self.steering_vector) < self.distance_before_swap:

            return PathingState(
                self.analyser, self.analyser.create_path(), self.steering
            )

        return self


class DeliveryPointDeliveringState(State):
    def __init__(self, analyser: Analyse, steering: SteeringUtils):
        super().__init__(analyser, steering)
        self.path = []
        self.is_close_to_delivery_point = False

    def on_frame(self):
        self.steering_vector = self.analyser.dropoff_coords - self.analyser.robot_pos
        signed_angle_degree = math.degrees(
            angle_between_vectors_signed(
                self.analyser.robot_vector, self.steering_vector
            )
        )
        self.path = [self.analyser.dropoff_coords]

        if self.analyser.are_coordinates_close(self.steering_vector, 40):
            self.is_close_to_delivery_point = True

        self.goal_vector_degrees = math.degrees(
            angle_between_vectors_signed(
                self.analyser.robot_vector,
                self.analyser.dropoff_coords - self.analyser.robot_pos,
            )
        )
        if self.is_close_to_delivery_point:
            print("CLOSE TO DELP")
            if abs(self.goal_vector_degrees) > 1:
                self.steering.stop()
                print("CLOSE AND TURNING")
                self.steering.turn(-1 * self.goal_vector_degrees, 3)
            print(self.goal_vector_degrees, self.analyser.robot_vector)
        else:
            self.steering.move_corrected(signed_angle_degree, 4, turn_speed=15)

    def swap_state(self):
        if self.is_close_to_delivery_point and abs(self.goal_vector_degrees) < 4:
            # stop
            self.steering.stop()
            return ReleaseBallsState(self.analyser, self.steering)
        return self


class ReleaseBallsState(State):
    def __init__(self, analyser: Analyse, steering: SteeringUtils):
        super().__init__(analyser, steering)
        self.timeout = 10

    def on_frame(self):
        self.steering.reverse_belt()
        self.steering.stop()
        self.steering.move_corrected(0, 0, turn_speed=0)

    def swap_state(self):
        # if self.timeout < time.time() - self.start_time:
        #     return PathingState(
        #         self.analyser, self.analyser.create_path(), self.steering
        #     )
        return self
