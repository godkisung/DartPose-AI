import serial
import time

class ArduinoReceiver:
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Connected to Arduino on {self.port} at {self.baudrate} baud.")
        except serial.SerialException as e:
            print(f"Warning: Could not connect to Arduino: {e}")

    def listen_for_trigger(self):
        """
        Listens for the 'HIT,timestamp' string from the Arduino.
        Returns the timestamp if hit detected, else None.
        """
        if not self.ser:
            return None
        
        if self.ser.in_waiting > 0:
            line = self.ser.readline().decode('utf-8').strip()
            if line.startswith("HIT"):
                _, timestamp_ms = line.split(',')
                return int(timestamp_ms)
        return None
