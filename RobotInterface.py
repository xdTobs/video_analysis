import json
import math
import time
import socket
import traceback


class RobotInterface:
    def __init__(self, host: str, port: int, online: bool = True):
        self.online = online
        self.sock: socket.socket
        self.connected: bool = False
        self.host = host
        self.port = port
        self.retry_attempts: int = 3
        self.retry_delay: int = 2
        self.last_command_type: str = ""
        self.last_command_speed: float = 0
        self.last_command_value: float = 0

    def connect(self):
        if self.online == False:
            return
        attempts = 0
        while attempts < self.retry_attempts:
            try:
                print("Trying to connect to robot at", self.host, "on port", self.port)
                self.sock: socket.socket = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM
                )
                self.sock.connect((self.host, self.port))
                print("Connected to robot! :)")
                self.connected = True
                return
            except socket.error as e:
                traceback.print_exc()
                attempts += 1
                print(f"Attempt {attempts} failed: {e}")
                if attempts == self.retry_attempts:
                    raise ConnectionError(
                        f"Failed to connect to the robot after {self.retry_attempts} attempts"
                    ) from e
                time.sleep(self.retry_delay)
        raise ConnectionError(
            f"Failed to connect to the robot after {self.retry_attempts} attempts"
        )

    def disconnect(self):
        try:
            self.sock.close()
            print("Disconnected from the robot")
            self.connected = False
        except:
            traceback.print_exc()
            raise ConnectionError("Failed to disconnect from the robot")

    def send_command(self, command: str, value: float, speedPercentage: int):
        #Clamp speedPercentage to -100 -> 100
        speedPercentage = max(-100, min(100, speedPercentage))
        
        if self.online == False:
            return
        if not self.connected:
            raise ConnectionError("Not connected to the robot")
        try:
            #if self.last_command_type == command and abs(self.last_command_speed - speedPercentage) < 5 and abs(self.last_command_value - value) < 5:
            #    print("Command too similar to previous command, not sending it")
            #    print(f"Commands: {self.last_command_type} {command}, values: {self.last_command_value} {value}, speeds: {self.last_command_speed} {speedPercentage}")
            #    return

            data = {
                "command": command,
                "value": value,
                "speedPercentage": speedPercentage,
            }
            self.last_command_speed = speedPercentage
            self.last_command_value = value
            self.last_command_type = command
            serialized_data = json.dumps(data).encode()
            self.sock.sendall(serialized_data)
            print("Data sent", data)
            
        except:
            traceback.print_exc()
            raise DataSendError("Failed to send data")

    def receive_command(self) -> str:
        if self.online == False:
            return ""
        if not self.connected:
            raise ConnectionError("Not connected to the robot")
        try:
            print("Waiting for data")
            data = self.sock.recv(1024)
            if not data:
                raise DataReceiveError("No data received; the connection may be broken")
            data = data.decode()
            print("Data received", data)
            return data
        except socket.error as e:
            traceback.print_exc()
            print("Socket error:", e)
            raise DataReceiveError(f"Socket error occurred: {e}") from e
        except Exception as e:
            traceback.print_exc()
            print("Unexpected error:", e)
            raise DataReceiveError(f"An unexpected error occurred: {e}") from e


class DataReceiveError(Exception):
    def __init__(self, message="Failed to receive data", *args):
        super().__init__(message, *args)
        self.message = message


class DataSendError(Exception):
    def __init__(self, message="Failed to send data", *args):
        super().__init__(message, *args)
        self.message = message
