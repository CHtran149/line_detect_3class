import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import cv2
import numpy as np
import math
import time
from ultralytics import YOLO

class LineFollowerROS2(Node):
    def __init__(self, video_source=0):
        super().__init__('line_follower_node')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open video source")
            raise RuntimeError("Cannot open video source")

        self.v_fixed = 0.4
        self.turn_start_time = None
        self.state = "STRAIGHT"
        self.model = YOLO(r"/root/ros2_ws/src/line_detections/models/best.pt")

        # Timer để gọi hàm xử lý frame khoảng 10-15 Hz (100 ms)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def clamp(self, value, min_val, max_val):
        return max(min(value, max_val), min_val)

    def detect_line_direction(self, frame):
        results = self.model.predict(frame, imgsz=320, conf=0.5, verbose=False)[0]
        class_ids = [int(box.cls[0]) for box in results.boxes]
        if len(class_ids) == 0:
            return None
        return class_ids[0]

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("Video ended or cannot read frame")
            rclpy.shutdown()
            return

        frame_resized = cv2.resize(frame, (320, 240))
        class_id = self.detect_line_direction(frame_resized)

        if class_id == 0:  # left turn
            self.state = "TURN_LEFT_90"
            w = self.clamp(1.5, -math.pi/2, math.pi/2)  
            self.turn_start_time = time.time()
        elif class_id == 1:  # right turn
            self.state = "TURN_RIGHT_90"
            w = self.clamp(-1.5, -math.pi/2, math.pi/2)  
            self.turn_start_time = time.time()
        elif class_id == 2:  # straight
            self.state = "STRAIGHT"
            w = 0.0
            self.turn_start_time = None
        else:
            self.state = "UNKNOWN"
            w = 0.0

        # Tạo message Twist để gửi tốc độ
        twist = Twist()
        twist.linear.x = self.v_fixed
        twist.angular.z = w
        self.publisher_.publish(twist)

        self.get_logger().info(f"State: {self.state} | v: {twist.linear.x:.2f}, w: {twist.angular.z:.2f}")

        

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerROS2(video_source=0)  
    rclpy.spin(node)
    node.cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
