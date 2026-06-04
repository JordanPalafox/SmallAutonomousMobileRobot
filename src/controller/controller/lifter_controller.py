import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
from builtin_interfaces.msg import Duration


MAST_JOINT = 'mast_inner_joint'
CARRIAGE_JOINT = 'carriage_joint'

MAST_MAX = 0.10
CARRIAGE_MAX = 0.10
POSITION_TOLERANCE = 0.005   # 5 mm — close enough to trigger next stage
TRAVEL_TIME = 4.0             # seconds per stage


class LifterController(Node):

    def __init__(self):
        super().__init__('lifter_controller_node')

        self._traj_pub = self.create_publisher(
            JointTrajectory,
            '/lifter_controller/joint_trajectory',
            10
        )

        self._cmd_sub = self.create_subscription(
            String,
            '/lifter/command',
            self._command_cb,
            10
        )

        self._joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_states_cb,
            10
        )

        self._mast_pos = 0.0
        self._carriage_pos = 0.0

        # Pending second-stage target set after first stage completes
        self._pending_mast = None
        self._pending_carriage = None

        self._timer = self.create_timer(0.1, self._check_pending)

        self.get_logger().info(
            "Lifter controller ready. Publish 'raise' or 'lower' to /lifter/command"
        )

    # ------------------------------------------------------------------
    def _joint_states_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name == MAST_JOINT:
                self._mast_pos = pos
            elif name == CARRIAGE_JOINT:
                self._carriage_pos = pos

    # ------------------------------------------------------------------
    def _command_cb(self, msg: String):
        cmd = msg.data.strip().lower()

        if cmd == 'raise':
            self.get_logger().info('RAISE: moving inner mast up first')
            # Stage 1 — mast to max, carriage stays put
            self._send(MAST_MAX, self._carriage_pos, TRAVEL_TIME)
            # Stage 2 — after mast reaches top, raise carriage
            self._pending_mast = MAST_MAX
            self._pending_carriage = CARRIAGE_MAX

        elif cmd == 'lower':
            self.get_logger().info('LOWER: lowering carriage first')
            # Stage 1 — carriage to 0, mast stays put
            self._send(self._mast_pos, 0.0, TRAVEL_TIME)
            # Stage 2 — after carriage is down, lower mast
            self._pending_mast = 0.0
            self._pending_carriage = 0.0

        elif cmd == 'stop':
            self.get_logger().info('STOP: holding current position')
            self._pending_mast = None
            self._pending_carriage = None
            self._send(self._mast_pos, self._carriage_pos, 0.5)

        else:
            self.get_logger().warn(f"Unknown command: '{cmd}'. Use raise / lower / stop")

    # ------------------------------------------------------------------
    def _check_pending(self):
        if self._pending_mast is None:
            return

        target_mast = self._pending_mast
        target_carriage = self._pending_carriage

        if target_carriage == CARRIAGE_MAX:
            # Waiting for mast to reach top before raising carriage
            if abs(self._mast_pos - MAST_MAX) < POSITION_TOLERANCE:
                self.get_logger().info('Mast at top — raising carriage')
                self._send(MAST_MAX, CARRIAGE_MAX, TRAVEL_TIME)
                self._pending_mast = None
                self._pending_carriage = None

        elif target_mast == 0.0:
            # Waiting for carriage to reach bottom before lowering mast
            if abs(self._carriage_pos - 0.0) < POSITION_TOLERANCE:
                self.get_logger().info('Carriage at bottom — lowering mast')
                self._send(0.0, 0.0, TRAVEL_TIME)
                self._pending_mast = None
                self._pending_carriage = None

    # ------------------------------------------------------------------
    def _send(self, mast: float, carriage: float, duration_sec: float):
        msg = JointTrajectory()
        msg.joint_names = [MAST_JOINT, CARRIAGE_JOINT]

        pt = JointTrajectoryPoint()
        pt.positions = [mast, carriage]
        pt.velocities = [0.0, 0.0]
        secs = int(duration_sec)
        nsecs = int((duration_sec - secs) * 1e9)
        pt.time_from_start = Duration(sec=secs, nanosec=nsecs)

        msg.points = [pt]
        self._traj_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LifterController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
