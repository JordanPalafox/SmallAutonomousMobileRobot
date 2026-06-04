#!/usr/bin/env python3
"""
Suscribe a /map y /slam_pose, renderiza con matplotlib y guarda PNG.

Uso:
  python3 plot_map_pose.py                          # guarda al recibir 1 mapa
  python3 plot_map_pose.py --out docs/img/pose_2d_map.png
  python3 plot_map_pose.py --live                   # ventana interactiva
  python3 plot_map_pose.py --pose-topic /mcl_pose   # cambiar topic de pose
"""
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped


def yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class MapPosePlotter(Node):
    def __init__(self, out_path, live, map_topic, pose_topic):
        super().__init__('map_pose_plotter')
        self.out_path = out_path
        self.live = live

        self.map_msg = None
        self.pose = None  # (x, y, theta)
        self.trajectory = []
        self.saved = False

        self.create_subscription(OccupancyGrid, map_topic, self.map_cb, 1)
        self.create_subscription(PoseStamped, pose_topic, self.pose_cb, 10)

        self.get_logger().info(
            f'Escuchando {map_topic} y {pose_topic}. '
            f'{"Modo live." if live else f"Guardado en {out_path}."}')

        if live:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.fig.canvas.manager.set_window_title('Map + Pose')

    # ── Callbacks ──────────────────────────────────────────────────────

    def map_cb(self, msg):
        self.map_msg = msg
        if self.live:
            self.render()
        elif self.pose is not None and not self.saved:
            self.render_and_save()

    def pose_cb(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        th = yaw_from_quat(msg.pose.orientation)
        self.pose = (x, y, th)
        self.trajectory.append((x, y))
        # No saturar memoria
        if len(self.trajectory) > 5000:
            self.trajectory = self.trajectory[-5000:]

        if self.live and self.map_msg is not None:
            self.render()
        elif self.map_msg is not None and not self.saved:
            self.render_and_save()

    # ── Render ─────────────────────────────────────────────────────────

    def render(self):
        self.ax.clear()
        self._draw(self.ax)
        plt.pause(0.001)

    def render_and_save(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        self._draw(ax)
        fig.tight_layout()
        fig.savefig(self.out_path, dpi=150, bbox_inches='tight')
        self.get_logger().info(f'Guardado en {self.out_path}')
        plt.close(fig)
        self.saved = True
        rclpy.shutdown()

    def _draw(self, ax):
        m = self.map_msg
        w, h, res = m.info.width, m.info.height, m.info.resolution
        ox, oy = m.info.origin.position.x, m.info.origin.position.y

        grid = np.array(m.data, dtype=np.int8).reshape(h, w)

        # -1 desconocido (gris), 0 libre (blanco), 100 ocupado (negro)
        img = np.full_like(grid, 205, dtype=np.uint8)
        img[grid == 0] = 255
        img[grid == 100] = 0

        extent = [ox, ox + w * res, oy, oy + h * res]
        ax.imshow(img, cmap='gray', vmin=0, vmax=255,
                  origin='lower', extent=extent)

        # Trayectoria histórica
        if len(self.trajectory) > 1:
            xs, ys = zip(*self.trajectory)
            ax.plot(xs, ys, '-', color='tab:blue', linewidth=1.0,
                    alpha=0.7, label='trayectoria')

        # Pose actual
        if self.pose is not None:
            x, y, th = self.pose
            arrow_len = max(0.25, 5 * res)
            ax.add_patch(FancyArrow(
                x, y,
                arrow_len * math.cos(th),
                arrow_len * math.sin(th),
                width=arrow_len * 0.25,
                color='red',
                length_includes_head=True,
                zorder=10))
            ax.plot(x, y, 'o', color='red', markersize=5, zorder=11,
                    label='pose actual')

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Occupancy grid + pose SLAM')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='pose_2d_map.png',
                        help='ruta del PNG de salida')
    parser.add_argument('--live', action='store_true',
                        help='ventana interactiva en vez de guardar')
    parser.add_argument('--map-topic', default='/map')
    parser.add_argument('--pose-topic', default='/slam_pose')
    args = parser.parse_args()

    rclpy.init()
    node = MapPosePlotter(args.out, args.live,
                          args.map_topic, args.pose_topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
