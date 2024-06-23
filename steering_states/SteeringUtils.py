from video_analysis.RobotInterface import RobotInterface
import time

class SteeringUtils:
    def __init__(self, robot_interface : RobotInterface):
        self.robot_interface = robot_interface
        
    
    def move_corrected(self, signed_angle_degrees, angle_degrees, speed):
            print(f"angle to target {angle_degrees}")
            if angle_degrees < 1.5:
                self.robot_interface.send_command("move", 100, speed)
            elif 1.5 <= angle_degrees <= 20:
                self.robot_interface.send_command("move-corrected", -1 * signed_angle_degrees, speed)
                print(f"Signed angle degrees {signed_angle_degrees}")
            elif angle_degrees > 20:
                turn = signed_angle_degrees * -1 / 3
                self.robot_interface.send_command("turn", turn, 30)
    def get_near_ball(self, signed_angle_degrees, angle_degrees, dist_to_ball):
        self.move_corrected(signed_angle_degrees, angle_degrees)

    def collect_ball(self, signed_angle_degrees, angle_degrees, dist_to_ball):
        self.move_corrected(signed_angle_degrees, angle_degrees)

    def start_belt(self):
        self.robot_interface.send_command("belt", 0, speedPercentage=100)

    def stop_belt(self):
        self.robot_interface.send_command("belt", 0, speedPercentage=-100)
        #TODO cannot sleep
        time.sleep(5)
        self.robot_interface.send_command("belt", 0, speedPercentage=0)

    def disconnect(self):
        print("Disconnecting from robot")
        self.robot_interface.disconnect()
        return
            
