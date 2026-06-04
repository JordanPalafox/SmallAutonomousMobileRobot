#!/usr/bin/env python3
"""Nodo detector de marcadores ArUco.

Suscribe `/cam_img`, detecta los marcadores con `ArucoDetector` y publica:
    /aruco_poses  (geometry_msgs/PoseArray)   poses 6DOF en frame de la camara
    /aruco_ids    (std_msgs/Int32MultiArray)  IDs en el MISMO orden que poses
    /aruco_pose_node/debug_image (sensor_msgs/Image)  frame anotado (opcional)

Es el contrato de vision puro definido en CLAUDE.md (`/aruco_poses <- vision`).
Para localizacion global usa `aruco_localization`, que ademas estima la pose del
robot. En produccion corre UNO de los dos nodos (ambos detectan): este es el
ligero para consumidores que solo necesitan las poses crudas en frame camara.

NOTA: PoseArray no transporta los IDs, por eso se publica `/aruco_ids` en
paralelo, indexado igual que `poses[i] <-> ids[i]`.
"""
from __future__ import annotations

import os

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray

from .aruco_detector import ArucoDetector
from .image_decode import imgmsg_to_bgr


def _load_camera_yaml(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    cam = data['camera']
    K = np.array(cam['camera_matrix'], dtype=np.float64).reshape(3, 3)
    dist = np.array(cam['distortion_coefficients'], dtype=np.float64).flatten()
    return K, dist


class ArucoPoseNode(Node):
    def __init__(self) -> None:
        super().__init__('aruco_pose_node')

        self.declare_parameter('image_topic', '/cam_img')
        self.declare_parameter('qos', 'sensor_data')
        self.declare_parameter('camera_params', 'src/perception/config/camera_params.yaml')
        self.declare_parameter('marker_length', 0.09)
        self.declare_parameter('dictionary', 'original')
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('publish_debug_image', True)

        self._image_topic = str(self.get_parameter('image_topic').value)
        qos_name = str(self.get_parameter('qos').value).lower()
        cam_path = str(self.get_parameter('camera_params').value)
        marker_length = float(self.get_parameter('marker_length').value)
        dictionary = str(self.get_parameter('dictionary').value)
        self._frame_id = str(self.get_parameter('frame_id').value)
        self._publish_debug = bool(self.get_parameter('publish_debug_image').value)

        if not os.path.isabs(cam_path):
            cam_path = os.path.abspath(cam_path)
        if not os.path.exists(cam_path):
            self.get_logger().error(f'No existe camera_params: {cam_path}')
            raise SystemExit(1)
        K, dist = _load_camera_yaml(cam_path)

        self._detector = ArucoDetector(
            camera_matrix=K, dist_coeffs=dist,
            marker_length=marker_length, dictionary=dictionary, refine=True,
        )
        self.get_logger().info(
            f'ArucoDetector dict={self._detector.dictionary} '
            f'marker={marker_length * 100:.1f}cm'
        )

        if qos_name in ('reliable', 'default'):
            img_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST, depth=10,
                durability=DurabilityPolicy.VOLATILE,
            )
        else:
            img_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST, depth=5,
                durability=DurabilityPolicy.VOLATILE,
            )

        self._bridge = CvBridge()
        self.create_subscription(Image, self._image_topic, self._image_cb, img_qos)
        self._pub_poses = self.create_publisher(PoseArray, '/aruco_poses', 10)
        self._pub_ids = self.create_publisher(Int32MultiArray, '/aruco_ids', 10)
        self._pub_debug = (
            self.create_publisher(Image, '/aruco_pose_node/debug_image', 10)
            if self._publish_debug else None
        )

        self.get_logger().info(f'aruco_pose_node listo en {self._image_topic}')

    def _image_cb(self, msg: Image) -> None:
        frame = imgmsg_to_bgr(self._bridge, msg)
        if frame is None:
            return

        dets = self._detector.detect(frame)

        pa = PoseArray()
        pa.header.stamp = msg.header.stamp
        pa.header.frame_id = self._frame_id
        ids = Int32MultiArray()
        for det in dets:
            pa.poses.append(det['pose'])
            ids.data.append(int(det['id']))
        self._pub_poses.publish(pa)
        self._pub_ids.publish(ids)

        if self._pub_debug is not None:
            annotated = self._detector.draw_detections(frame, dets)
            out = self._bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out.header = pa.header
            self._pub_debug.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArucoPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
