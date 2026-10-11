#!/usr/bin/env python3
"""
Pipeline Verification Script
Publishes synthetic RGB-D camera frames into ROS 2 topics and inspects
the resulting /target_grasp_pose and /tag_net/status messages.
"""

import sys
import time
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Header
from cv_bridge import CvBridge


class PipelineTester(Node):
    def __init__(self):
        super().__init__('pipeline_tester')
        self.bridge = CvBridge()

        # Publishers
        self.rgb_pub = self.create_publisher(Image, '/camera/image', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth_image', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        self.task_pub = self.create_publisher(String, '/task_command', 10)

        # Subscribers
        self.pose_received = False
        self.pose_sub = self.create_subscription(
            PoseStamped, '/target_grasp_pose', self.pose_callback, 10
        )
        self.status_sub = self.create_subscription(
            String, '/tag_net/status', self.status_callback, 10
        )

        self.timer = self.create_timer(0.5, self.publish_synthetic_frame)
        self.get_logger().info('Pipeline Tester Node running. Publishing synthetic RGB-D frames...')

    def pose_callback(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        self.get_logger().info(f'[PERCEPTION FEEDBACK] Pose Received: X={p.x:.3f}, Y={p.y:.3f}, Z={p.z:.3f} | Qz={q.z:.3f}, Qw={q.w:.3f}')
        self.pose_received = True

    def status_callback(self, msg: String):
        self.get_logger().info(f'[TELEMETRY] {msg.data}')

    def publish_synthetic_frame(self):
        # 480x640 synthetic image with a blue handle and orange tool tip
        img = np.full((480, 640, 3), 40, dtype=np.uint8)
        # Handle (blue cylinder)
        import cv2
        cv2.rectangle(img, (200, 220), (320, 260), (220, 100, 30), -1)
        # Head (orange block)
        cv2.rectangle(img, (320, 210), (440, 270), (30, 120, 230), -1)

        # Depth map (table at 0.7m, tool at 0.65m)
        depth = np.full((480, 640), 0.70, dtype=np.float32)
        depth[210:270, 200:440] = 0.65

        hdr = Header()
        hdr.stamp = self.get_clock().now().to_msg()
        hdr.frame_id = 'camera_link'

        rgb_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        rgb_msg.header = hdr

        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        depth_msg.header = hdr

        info_msg = CameraInfo()
        info_msg.header = hdr
        info_msg.k = [554.25, 0.0, 320.0, 0.0, 554.25, 240.0, 0.0, 0.0, 1.0]

        task_msg = String()
        task_msg.data = 'handover'

        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.info_pub.publish(info_msg)
        self.task_pub.publish(task_msg)


def main():
    rclpy.init()
    tester = PipelineTester()
    start_time = time.time()
    try:
        while rclpy.ok() and (time.time() - start_time < 5.0):
            rclpy.spin_once(tester, timeout_sec=0.1)
            if tester.pose_received:
                tester.get_logger().info('>>> SUCCESS: Full ROS 2 end-to-end perception loop verified!')
                break
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
