#!/usr/bin/env python3
import os
from pathlib import Path

import yaml
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Trigger


class MapSaverNode(Node):

    def __init__(self):
        super().__init__('map_saver')
        self.declare_parameter('map_path', os.path.expanduser('~/ros2_maps/warehouse'))
        # auto_save OFF by default: auto-saving the live /map on Ctrl+C would
        # clobber a good saved map with whatever's live at shutdown (e.g. a
        # fresh/empty grid right after a slam restart). Save explicitly instead.
        self.declare_parameter('auto_save', False)

        self._latest_map: OccupancyGrid | None = None
        self.create_subscription(OccupancyGrid, '/map', self._map_cb, 1)
        self.create_service(Trigger, '~/save_map', self._save_cb)

        save_path = self.get_parameter('map_path').value
        auto = bool(self.get_parameter('auto_save').value)
        self.get_logger().info(
            f'MapSaver ready — maps will be saved to {save_path}.pgm / .yaml\n'
            f'  Trigger manually:  ros2 service call /map_saver/save_map std_srvs/srv/Trigger {{}}\n'
            f'  Auto-save on Ctrl+C: {"ON" if auto else "OFF (set auto_save:=true to enable)"}'
        )

    def _map_cb(self, msg: OccupancyGrid):
        self._latest_map = msg

    def _save_cb(self, _request, response):
        if self._latest_map is None:
            response.success = False
            response.message = 'No map received yet — drive the robot first'
            return response
        path = self.get_parameter('map_path').value
        try:
            self._write_map(self._latest_map, path)
            response.success = True
            response.message = f'Saved to {path}.pgm / {path}.yaml'
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _write_map(self, msg: OccupancyGrid, base_path: str):
        base_path = os.path.expanduser(base_path)
        Path(base_path).parent.mkdir(parents=True, exist_ok=True)

        w   = msg.info.width
        h   = msg.info.height
        res = msg.info.resolution
        ox  = msg.info.origin.position.x
        oy  = msg.info.origin.position.y

        # nav2 map_server pixel convention:
        #   254 = free (white), 0 = occupied (black), 205 = unknown (gray)
        pixels = bytearray(w * h)
        for i, val in enumerate(msg.data):
            if val == 0:
                pixels[i] = 254   # free
            elif val == 100:
                pixels[i] = 0     # occupied
            else:
                pixels[i] = 205   # unknown (-1)

        # ROS OccupancyGrid origin is bottom-left; PGM row 0 is top-left → flip
        rows = [pixels[r * w:(r + 1) * w] for r in range(h)]
        rows.reverse()
        pgm_data = bytearray()
        for row in rows:
            pgm_data.extend(row)

        pgm_path = base_path + '.pgm'
        with open(pgm_path, 'wb') as f:
            f.write(f'P5\n{w} {h}\n255\n'.encode())
            f.write(pgm_data)

        yaml_path = base_path + '.yaml'
        meta = {
            'image': os.path.basename(pgm_path),
            'resolution': float(res),
            'origin': [float(ox), float(oy), 0.0],
            'negate': 0,
            'occupied_thresh': 0.65,
            'free_thresh': 0.196,
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(meta, f, default_flow_style=False)

        self.get_logger().info(f'Map saved — {w}x{h} cells @ {res:.3f} m/cell')
        self.get_logger().info(f'  {pgm_path}')
        self.get_logger().info(f'  {yaml_path}')

    def destroy_node(self):
        if self._latest_map is not None and bool(self.get_parameter('auto_save').value):
            path = self.get_parameter('map_path').value
            self.get_logger().info('Auto-saving map on shutdown…')
            try:
                self._write_map(self._latest_map, path)
            except Exception as exc:
                self.get_logger().error(f'Auto-save failed: {exc}')
        super().destroy_node()


def main():
    rclpy.init()
    node = MapSaverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
