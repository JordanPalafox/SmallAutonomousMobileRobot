"""
DashboardNode — ROS2 node that hosts the Flask web dashboard.

The Flask + SocketIO server runs in a daemon thread so the ROS2
executor can spin normally in the main thread.
"""

import threading

import rclpy
from rclpy.node import Node

from dashboard.ros_bridge import RosBridge
import dashboard.app as flask_app


class DashboardNode(Node):
    def __init__(self) -> None:
        super().__init__("dashboard_node")

        # Declare parameters with defaults
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8080)
        self.declare_parameter("debug", False)

        host = self.get_parameter("host").get_parameter_value().string_value
        port = self.get_parameter("port").get_parameter_value().integer_value
        debug = self.get_parameter("debug").get_parameter_value().bool_value

        self.get_logger().info(
            f"Dashboard starting on http://{host}:{port}"
        )

        # Wire the RosBridge into the Flask app module
        self._bridge = RosBridge(self)
        flask_app.ros_bridge = self._bridge

        # Start the SocketIO background emitter thread
        flask_app._start_background_emitter()

        # Start the Flask/SocketIO server in a daemon thread so that
        # rclpy.spin() can run in the main thread as required.
        server_thread = threading.Thread(
            target=self._run_server,
            args=(host, port, debug),
            daemon=True,
        )
        server_thread.start()

    def _run_server(self, host: str, port: int, debug: bool) -> None:
        try:
            flask_app.socketio.run(
                flask_app.app,
                host=host,
                port=port,
                debug=debug,
                use_reloader=False,  # reloader is incompatible with threads
                log_output=debug,
                allow_unsafe_werkzeug=True,
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Flask server error: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DashboardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
