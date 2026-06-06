#!/usr/bin/env python3
"""gemelo_mirror — espejo en vivo del robot real dentro del gemelo de Gazebo.

Suscribe la pose del robot real (`/slam_pose`, frame `map`) y la fija sobre el
modelo del robot en el mundo Gazebo (`almacen_racks`) llamando al servicio
Ignition `/world/<world>/set_pose`. Así el robot del gemelo SIGUE en vivo al
robot físico — es un visualizador, no una simulación con físicas.

Transformación map -> mundo Gazebo (rígida SE(2)):
    Después del ancla ArUco el frame `map` del SLAM tiene su origen en la
    esquina SW del almacén REAL (x=E-W 0→3.65 m, y=N-S 0→4.85 m).
    El mundo Gazebo tiene su origen en la esquina SE (x=N-S 0→4.85 m,
    y=E-W east→west 0→3.65 m).  La relación es:
        gz_x = map_y + tx        (norte en Gazebo = norte en REAL)
        gz_y = −map_x + ty       (oeste en Gazebo = ancho − este en REAL)
    con yaw_gz = yaw_map − 90°.
    Equivalente matricial: gz = R(-90°) · map_xy + (tx, ty)
    Defaults para pista 3.65×4.85 m con ancla ArUco en esquina SW:
        yaw = -90°, t = (0.0, 3.65)
"""
from __future__ import annotations

import math
import shutil
import subprocess

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


def _yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class GemeloMirror(Node):
    def __init__(self) -> None:
        super().__init__('gemelo_mirror')

        self.declare_parameter('world_name', 'almacen_racks')
        self.declare_parameter('model_name', 'puzzlebot')
        self.declare_parameter('pose_topic', '/slam_pose')
        self.declare_parameter('rate', 15.0)
        # Transformacion map -> mundo gemelo
        self.declare_parameter('world_from_map_yaw_deg', -90.0)
        self.declare_parameter('world_from_map_xy', [0.0, 3.65])
        # Altura a la que se dibuja el robot en el gemelo [m]
        self.declare_parameter('z', 0.0)

        self._world = str(self.get_parameter('world_name').value)
        self._model = str(self.get_parameter('model_name').value)
        topic = str(self.get_parameter('pose_topic').value)
        rate = float(self.get_parameter('rate').value)
        self._yaw = math.radians(float(self.get_parameter('world_from_map_yaw_deg').value))
        txy = [float(v) for v in self.get_parameter('world_from_map_xy').value]
        self._tx, self._ty = txy[0], txy[1]
        self._z = float(self.get_parameter('z').value)

        # Binario de servicio Ignition / Gazebo
        self._svc = 'ign' if shutil.which('ign') else ('gz' if shutil.which('gz') else None)
        if self._svc is None:
            self.get_logger().error('No encuentro `ign` ni `gz` en PATH; no puedo fijar la pose.')

        self._latest: PoseStamped | None = None
        self.create_subscription(PoseStamped, topic, self._on_pose, 10)
        self.create_timer(1.0 / max(1.0, rate), self._tick)
        self.get_logger().info(
            f'gemelo_mirror: {topic} -> set_pose({self._model}@{self._world}) '
            f'@ {rate:.0f} Hz, T=({self._tx:.3f},{self._ty:.3f}, yaw={math.degrees(self._yaw):.0f})')

    def _on_pose(self, msg: PoseStamped) -> None:
        self._latest = msg

    def _tick(self) -> None:
        if self._latest is None or self._svc is None:
            return
        p = self._latest.pose
        mx, my = p.position.x, p.position.y
        myaw = _yaw_from_quat(p.orientation)
        # map -> gemelo
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        gx = c * mx - s * my + self._tx
        gy = s * mx + c * my + self._ty
        gyaw = myaw + self._yaw
        qz, qw = math.sin(gyaw / 2.0), math.cos(gyaw / 2.0)

        req = (f'name: "{self._model}", '
               f'position: {{x: {gx:.4f}, y: {gy:.4f}, z: {self._z:.4f}}}, '
               f'orientation: {{x: 0, y: 0, z: {qz:.6f}, w: {qw:.6f}}}')
        cmd = [self._svc, 'service', '-s', f'/world/{self._world}/set_pose',
               '--reqtype', 'ignition.msgs.Pose',
               '--reptype', 'ignition.msgs.Boolean',
               '--timeout', '300', '--req', req]
        try:
            subprocess.run(cmd, capture_output=True, timeout=1.0)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'set_pose falló: {e}', throttle_duration_sec=5.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GemeloMirror()
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
