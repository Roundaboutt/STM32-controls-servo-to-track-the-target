import serial
import math

class Control:
    def __init__(self):
        self.ser = serial.Serial('COM3', 115200)

        self.kp = 0.45
        self.ki = 0.02
        self.kd = 0.15

        self.last_error_x = 0
        self.integral_x = 0
        self.last_error_y = 0
        self.integral_y = 0

        self.Servo_data = [0, 0]

        self.HEADER = b'\xFF'
        self.FOOTER = b'\xFE'

    def send_packet(self, data):
        data_bytes = bytes(data)
        packet = self.HEADER + data_bytes + self.FOOTER
        self.ser.write(packet)

    def pid_control(self, error, last_error, integral, kp, ki, kd):
        proportional = kp * error
        integral += error
        integral_term = ki * integral
        derivative = kd * (error - last_error)
        output = proportional + integral_term + derivative
        last_error = error
        return output, last_error, integral

    def output_to_servo(self, output_x, output_y):
        Servo_dx = int(math.atan2(output_x * 2.2/66, 12) * 180 / math.pi)
        Servo_dy = int(math.atan2(output_y * 2.2/66, 12) * 180 / math.pi)
        return Servo_dx + 128, Servo_dy + 128

    def send_to_stm32(self, cx, cy):
        img_center_x = 641/2
        img_center_y = 479/2

        error_x = img_center_x - cx
        error_y = img_center_y - cy

        output_x, self.last_error_x, self.integral_x = self.pid_control(
            error_x, self.last_error_x, self.integral_x, self.kp, self.ki, self.kd)
        output_y, self.last_error_y, self.integral_y = self.pid_control(
            error_y, self.last_error_y, self.integral_y, self.kp, self.ki, self.kd)

        self.Servo_data[0], self.Servo_data[1] = self.output_to_servo(output_x, output_y)
        self.send_packet(self.Servo_data)
        print(f"Send({self.Servo_data[0]},{self.Servo_data[1]}) to STM32")