import math
from RobotInterface import RobotInterface
import time
from utils import angle_between_vectors, angle_between_vectors_signed


class SteeringUtils:
    def __init__(self, robot_interface: RobotInterface):
        self.robot_interface = robot_interface

    def move_corrected(self, signed_angle_degrees, speed, turn_speed=30, state=None, turn_speed_turning=15):
        angle_degrees = abs(signed_angle_degrees)
        if angle_degrees < 1.5:
            self.robot_interface.send_command("move", 20, speed, state=state)
        elif 1.5 <= angle_degrees <= 20:
            self.robot_interface.send_command(
                "move-corrected", -1/3 * signed_angle_degrees, speed, state=state
            )
        elif angle_degrees > 20:
            turn = signed_angle_degrees * -1 / 3
            self.robot_interface.send_command("turn", turn, turn_speed_turning, state)

    def turn(self, angle_degrees, speed, state=None):
        self.robot_interface.send_command("turn", angle_degrees, speed, state=state)


    def stop(self):
        self.robot_interface.send_command("stop", 0, 0)

    def get_near_ball(self, signed_angle_degrees, angle_degrees, dist_to_ball):
        self.move_corrected(signed_angle_degrees, angle_degrees)

    def collect_ball(self, signed_angle_degrees, angle_degrees, dist_to_ball):
        self.move_corrected(signed_angle_degrees, angle_degrees)

    def start_belt(self):
        self.robot_interface.send_command("belt", 0, speedPercentage=100)

    def stop_belt(self):
        self.robot_interface.send_command("belt", 0, speedPercentage=0)

    def reverse_belt(self):
        self.robot_interface.send_command("belt", 0, speedPercentage=-100)

    def disconnect(self):
        print("Disconnecting from robot")
        self.robot_interface.disconnect()
        return
