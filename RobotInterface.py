import json
import time
import socket
import sys

class RobotInterface:
    def __init__(self, host : str, port : int, online : bool = True):
        self.online = online
        self.sock : socket.socket
        self.connected : bool = False
        self.host = host
        self.port = port
        self.retry_attempts : int = 3
        self.retry_delay : int = 2
        
        
    def connect(self):
        if self.online == False:
            return
        attempts = 0
        while attempts < self.retry_attempts:
            try:
                self.sock : socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                print("Connected to the robot at", self.host, "on port", self.port)
                self.connected = True
                return
            except socket.error as e:
                attempts += 1
                print(f"Attempt {attempts} failed: {e}")
                if attempts == self.retry_attempts:
                    raise ConnectionError(f"Failed to connect to the robot after {self.retry_attempts} attempts") from e
                time.sleep(self.retry_delay)
        raise ConnectionError(f"Failed to connect to the robot after {self.retry_attempts} attempts")
    
    def disconnect(self):
        try:
            self.sock.close()
            print("Disconnected from the robot")
            self.connected = False
        except:
            raise ConnectionError("Failed to disconnect from the robot")

    def send_command(self, command : str, value : float,speedPercentage: int):
        if self.online == False:
            return
        if not self.connected:
            raise ConnectionError("Not connected to the robot")
        try:
            data = {
            "command": command,
            "value": value,
            "speedPercentage":speedPercentage
            }
            serialized_data = json.dumps(data).encode()
            self.sock.sendall(serialized_data)
            print("Data sent",data)
        except:
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
            print("Data received",data)
            return data
        except socket.error as e:
            print("Socket error:", e)
            raise DataReceiveError(f"Socket error occurred: {e}") from e
        except Exception as e:
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