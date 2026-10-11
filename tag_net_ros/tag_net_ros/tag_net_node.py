#!/usr/bin/env python3
"""
TAG-Net ROS 2 Perception & Grasp Synthesis Node
Subscribes to RGB-D camera sensor streams from Gazebo Harmonic,
executes high-speed CUDA inference on RTX 5070 Ti,
and publishes 6-DoF task-aware grasp poses to the manipulation pipeline.
"""

import os
import sys
import time
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Header
from cv_bridge import CvBridge

# Ensure models directory is discoverable
candidate_paths = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
    '/mnt/c/Users/hareh/.gemini/antigravity/scratch/tag_grasp_net',
    'C:/Users/hareh/.gemini/antigravity/scratch/tag_grasp_net'
]
PROJECT_ROOT = candidate_paths[0]
for p in candidate_paths:
    if os.path.exists(os.path.join(p, 'models')):
        PROJECT_ROOT = p
        break

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import torch
    from models.tag_net import TAGNet
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"Warning: Failed to import torch/TAGNet: {e}")
    TORCH_AVAILABLE = False


class TAGNetNode(Node):
    def __init__(self):
        super().__init__('tag_net_node')
        self.get_logger().info('Initializing TAG-Net ROS 2 Node...')

        # Parameters
        self.declare_parameter('checkpoint_path', os.path.join(PROJECT_ROOT, 'checkpoints', 'tag_net_best.pth'))
        self.declare_parameter('default_task_id', 1)  # 1 = tool_use, 0 = handover/inspection
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('camera_frame', 'camera_link')

        self.checkpoint_path = self.get_parameter('checkpoint_path').value
        self.current_task_id = self.get_parameter('default_task_id').value
        self.conf_thresh = self.get_parameter('confidence_threshold').value
        self.camera_frame = self.get_parameter('camera_frame').value

        # Bridge & State
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_intrinsics = None

        # Device Selection (RTX 5070 Ti CUDA with graceful fallback)
        self.device = torch.device('cpu')
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                test_t = torch.zeros(1, device='cuda')
                _ = test_t + 1
                self.device = torch.device('cuda')
                self.get_logger().info(f'Perception Hardware Backend: CUDA on {torch.cuda.get_device_name(0)}')
            except Exception as e:
                self.get_logger().warn(f'CUDA available but device execution fallback to CPU (SM_120 compatibility): {e}')
                self.device = torch.device('cpu')
        else:
            self.get_logger().info('Perception Hardware Backend: CPU')

        # Load PyTorch Model
        self.model = None
        self.load_model()

        # QoS for sensor topics
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/image', self.rgb_callback, sensor_qos
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth_image', self.depth_callback, sensor_qos
        )
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, sensor_qos
        )
        self.task_sub = self.create_subscription(
            String, '/task_command', self.task_callback, 10
        )

        # Publishers
        self.grasp_pub = self.create_publisher(PoseStamped, '/target_grasp_pose', 10)
        self.vis_pub = self.create_publisher(Image, '/tag_net/grasp_visualization', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/tag_net/markers', 10)
        self.status_pub = self.create_publisher(String, '/tag_net/status', 10)

        # Perception Evaluation Timer (runs at 10 Hz)
        self.timer = self.create_timer(0.1, self.inference_cycle)
        self.get_logger().info('TAG-Net Node successfully initialized and listening for camera streams!')

    def load_model(self):
        if not TORCH_AVAILABLE:
            self.get_logger().warn('PyTorch not yet loaded. Running in mock perception mode.')
            return

        try:
            self.model = TAGNet(in_channels=4, base_channels=32).to(self.device)
            if os.path.exists(self.checkpoint_path):
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                state_dict = checkpoint['model_state_dict'] if (isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint) else checkpoint
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.get_logger().info(f'Successfully loaded TAG-Net weights from {self.checkpoint_path}')
            else:
                self.get_logger().warn(f'Checkpoint not found at {self.checkpoint_path}. Using initial weights.')
        except Exception as e:
            self.get_logger().error(f'Failed to load TAG-Net model: {e}')
            self.model = None

    def rgb_callback(self, msg: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB conversion error: {e}')

    def depth_callback(self, msg: Image):
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.latest_depth = np.nan_to_num(depth_img, nan=1.0, posinf=2.0, neginf=0.0)
        except Exception as e:
            self.get_logger().error(f'Depth conversion error: {e}')

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5]
            }
            self.get_logger().info(f'Camera Intrinsics Acquired: fx={msg.k[0]:.1f}, fy={msg.k[4]:.1f}, cx={msg.k[2]:.1f}, cy={msg.k[5]:.1f}')

    def task_callback(self, msg: String):
        command = msg.data.lower()
        if 'tool' in command:
            self.current_task_id = 1
            self.get_logger().info('Task Context Updated: TOOL_USE (Affordance Gate Active for Functional Head)')
        elif 'hand' in command:
            self.current_task_id = 0
            self.get_logger().info('Task Context Updated: HANDOVER (Affordance Gate Active for Handle Grip)')
        elif 'inspect' in command:
            self.current_task_id = 2
            self.get_logger().info('Task Context Updated: INSPECTION (Safe Non-Occluding Grip)')
        else:
            try:
                self.current_task_id = int(command)
            except ValueError:
                self.get_logger().warn(f'Unknown task command "{command}". Defaulting to task_id={self.current_task_id}')

    def inference_cycle(self):
        if self.latest_rgb is None or self.latest_depth is None:
            return

        t_start = time.time()
        H_orig, W_orig = self.latest_rgb.shape[:2]

        import cv2
        # Resize to network resolution 224x224
        rgb_resized = cv2.resize(self.latest_rgb, (224, 224))
        depth_resized = cv2.resize(self.latest_depth, (224, 224))

        # Normalize
        rgb_norm = (rgb_resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        depth_norm = np.clip(depth_resized.astype(np.float32), 0.0, 2.0) / 2.0

        # Construct 4-channel tensor: [B, 4, H, W]
        rgbd = np.concatenate([rgb_norm, np.expand_dims(depth_norm, axis=-1)], axis=-1)
        rgbd_tensor = torch.from_numpy(rgbd.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        if self.model is not None:
            with torch.no_grad():
                out = self.model(rgbd_tensor)
                qual = out['quality_geom'].squeeze().cpu().numpy()
                cos_2th = out['cos_2theta'].squeeze().cpu().numpy()
                sin_2th = out['sin_2theta'].squeeze().cpu().numpy()
                width = out['width'].squeeze().cpu().numpy()
                affordance = out['affordance_mask'].squeeze().cpu().numpy()
                gated_score = out['quality_task'].squeeze().cpu().numpy()
        else:
            # Fallback mock perception for test
            gated_score = np.ones((224, 224), dtype=np.float32) * 0.8
            cos_2th = np.ones((224, 224), dtype=np.float32)
            sin_2th = np.zeros((224, 224), dtype=np.float32)
            width = np.ones((224, 224), dtype=np.float32) * 0.08
            affordance = np.ones((224, 224), dtype=np.float32)

        inference_time_ms = (time.time() - t_start) * 1000.0

        # Find best grasp pose
        max_idx = np.unravel_index(np.argmax(gated_score), gated_score.shape)
        v_net, u_net = max_idx
        best_score = float(gated_score[v_net, u_net])

        # Map back to original image coordinates
        u_orig = int(u_net * W_orig / 224)
        v_orig = int(v_net * H_orig / 224)

        # Grasp orientation
        theta = 0.5 * math.atan2(float(sin_2th[v_net, u_net]), float(cos_2th[v_net, u_net]))
        grasp_width = float(width[v_net, u_net])

        # Depth & 3D unprojection
        Z = float(self.latest_depth[v_orig, u_orig])
        if Z <= 0.01 or Z > 3.0:
            Z = 0.65  # fallback default workspace depth

        fx = self.camera_intrinsics['fx'] if self.camera_intrinsics else 554.25
        fy = self.camera_intrinsics['fy'] if self.camera_intrinsics else 554.25
        cx = self.camera_intrinsics['cx'] if self.camera_intrinsics else W_orig / 2.0
        cy = self.camera_intrinsics['cy'] if self.camera_intrinsics else H_orig / 2.0

        X = (u_orig - cx) * Z / fx
        Y = (v_orig - cy) * Z / fy

        # Convert planar theta to Quaternion (top-down grasp approach along Z-axis)
        # Roll = 0, Pitch = 0, Yaw = theta
        qz = math.sin(theta / 2.0)
        qw = math.cos(theta / 2.0)

        # Publish Target Grasp Pose
        pose_msg = PoseStamped()
        pose_msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.camera_frame)
        pose_msg.pose.position = Point(x=float(X), y=float(Y), z=float(Z))
        pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=float(qz), w=float(qw))
        self.grasp_pub.publish(pose_msg)

        # Publish Visual Debug Overlay
        vis_img = self.latest_rgb.copy()
        cv2.circle(vis_img, (u_orig, v_orig), 7, (0, 0, 255), -1)

        # Draw grasp bounding box
        length = 40
        w_px = int(grasp_width * 200)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # 4 corners
        p1 = (int(u_orig - length*sin_t - w_px*cos_t), int(v_orig + length*cos_t - w_px*sin_t))
        p2 = (int(u_orig + length*sin_t - w_px*cos_t), int(v_orig - length*cos_t - w_px*sin_t))
        p3 = (int(u_orig + length*sin_t + w_px*cos_t), int(v_orig - length*cos_t + w_px*sin_t))
        p4 = (int(u_orig - length*sin_t + w_px*cos_t), int(v_orig + length*cos_t + w_px*sin_t))

        cv2.polylines(vis_img, [np.array([p1, p2, p3, p4], dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(vis_img, f'Task: {self.current_task_id} | TSR: {best_score:.2f} | Lat: {inference_time_ms:.1f}ms',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        try:
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding='bgr8')
            vis_msg.header = pose_msg.header
            self.vis_pub.publish(vis_msg)
        except Exception as e:
            self.get_logger().error(f'Visualizer bridge error: {e}')

        # Publish 3D Markers for RViz
        self.publish_rviz_markers(X, Y, Z, theta, grasp_width, pose_msg.header)

        # Publish status string
        status_msg = String()
        status_msg.data = f'{{"task_id": {self.current_task_id}, "score": {best_score:.3f}, "latency_ms": {inference_time_ms:.2f}, "x": {X:.3f}, "y": {Y:.3f}, "z": {Z:.3f}, "theta_deg": {math.degrees(theta):.1f}}}'
        self.status_pub.publish(status_msg)

    def publish_rviz_markers(self, x, y, z, theta, width, header):
        marker_array = MarkerArray()

        # Marker 1: Arrow indicating approach vector
        arrow = Marker()
        arrow.header = header
        arrow.ns = 'grasp_approach'
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.scale.x = 0.015  # shaft diameter
        arrow.scale.y = 0.03   # head diameter
        arrow.scale.z = 0.05   # head length
        arrow.color.r = 1.0
        arrow.color.g = 0.8
        arrow.color.b = 0.0
        arrow.color.a = 0.9

        # Start 15cm above, point to contact point
        arrow.points.append(Point(x=x, y=y, z=z - 0.15))
        arrow.points.append(Point(x=x, y=y, z=z))
        marker_array.markers.append(arrow)

        # Marker 2: Contact Center Sphere
        sphere = Marker()
        sphere.header = header
        sphere.ns = 'grasp_contact'
        sphere.id = 1
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position = Point(x=x, y=y, z=z)
        sphere.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        sphere.scale.x = 0.03
        sphere.scale.y = 0.03
        sphere.scale.z = 0.03
        sphere.color.r = 0.0
        sphere.color.g = 1.0
        sphere.color.b = 0.2
        sphere.color.a = 0.95
        marker_array.markers.append(sphere)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = TAGNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
