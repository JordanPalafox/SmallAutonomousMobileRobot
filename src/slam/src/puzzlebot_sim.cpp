// puzzlebot_sim.cpp — realistic 2D simulator for the Puzzlebot SLAM stack.
//
// Models the things that actually break SLAM on the real robot:
//   * WHEEL SLIP — the robot's true motion differs from what the
//     encoders report, especially in rotation, so /odom drifts.
//   * LIDAR NOISE — gaussian range noise + random beam dropouts +
//     the real RPLidar A1 cadence (~7 Hz) and 360-beam resolution.
//
// Environment: the 4.86 × 3.76 m track (outer walls) + three interior
// "rack" rectangles, all as line segments.  The LiDAR raycasts against
// them from the GROUND-TRUTH pose; odometry integrates a slipped,
// noisy version so the SLAM front-end has to fight the same drift it
// sees in reality.
//
// Publishes:
//   /scan                 (sensor_msgs/LaserScan, frame "laser", 7 Hz)
//   /odom                 (nav_msgs/Odometry, slipped, 30 Hz)
//   /ground_truth         (geometry_msgs/PoseStamped, true pose)
//   TF odom→base_link, static base_link→laser
// Subscribes:
//   /cmd_vel  and  /cmd_vel_in   (geometry_msgs/Twist)

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <random>
#include <cmath>
#include <vector>
#include <deque>

using namespace std::chrono_literals;

struct Seg { double x1, y1, x2, y2; };

class PuzzlebotSim : public rclcpp::Node {
public:
  PuzzlebotSim() : Node("puzzlebot_sim"), rng_(7) {
    // ── params ──
    room_x_ = declare_parameter("room_size_x", 4.86);
    room_y_ = declare_parameter("room_size_y", 3.76);
    // ── RPLidar A1M8 specifications (Slamtec datasheet) ──
    //   range 0.15–12 m · 10 Hz · ≤1° (360 pts) · accuracy 1% of dist
    lidar_hz_ = declare_parameter("lidar_hz", 10.0);                 // A1 sustains 10 Hz
    lidar_beams_ = declare_parameter("lidar_beams", 360);           // 1° resolution
    range_max_ = declare_parameter("lidar_range_max", 12.0);
    range_min_ = declare_parameter("lidar_range_min", 0.15);
    // A1 distance accuracy = 1% of range (distance-proportional), plus a
    // small fixed floor for close-range quantisation.
    range_noise_pct_ = declare_parameter("lidar_range_noise_pct", 0.01);   // 1%
    range_noise_floor_ = declare_parameter("lidar_range_noise_floor", 0.005); // 5 mm
    dropout_prob_ = declare_parameter("lidar_dropout_prob", 0.03);  // 3% beams
    // Wheel-slip model.
    slip_rot_ = declare_parameter("slip_rotation", 0.12);   // 12% rot slip σ
    slip_trans_ = declare_parameter("slip_translation", 0.04);
    odom_noise_v_ = declare_parameter("odom_noise_v", 0.01);
    odom_noise_w_ = declare_parameter("odom_noise_w", 0.02);
    laser_x_ = declare_parameter("laser_x", 0.04);
    laser_yaw_ = declare_parameter("laser_yaw", M_PI);   // A1 mounted backwards
    // ── LiDAR latency (the real-robot killer) ──
    // The scan is CAPTURED from the ground-truth pose at time T but is
    // only DELIVERED `lidar_latency` seconds later (WiFi / USB buffering).
    // stamp_on_publish models a driver that timestamps at delivery time
    // instead of capture time — that wrong stamp is what makes the scan
    // get matched against the WRONG odom pose → drift.
    lidar_latency_ = declare_parameter("lidar_latency", 0.15);          // s
    stamp_on_publish_ = declare_parameter("lidar_stamp_on_publish", false);

    buildEnvironment();

    // start pose = centre of room
    tx_ = 0.0; ty_ = 0.0; tth_ = 0.0;     // ground truth
    odx_ = 0.0; ody_ = 0.0; odth_ = 0.0;  // slipped odom

    scan_pub_ = create_publisher<sensor_msgs::msg::LaserScan>("/scan", rclcpp::SensorDataQoS());
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
    gt_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/ground_truth", 10);
    // Paths to make the slip VISIBLE: raw dead-reckoning odom drifts away
    // from the true trajectory; SLAM should track the true one.
    odom_path_pub_ = create_publisher<nav_msgs::msg::Path>("/odom_path", 1);
    gt_path_pub_ = create_publisher<nav_msgs::msg::Path>("/ground_truth_path", 1);
    odom_path_.header.frame_id = "map";
    gt_path_.header.frame_id = "map";
    tf_br_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    static_br_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
    publishStaticLaser();

    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", 10, [this](geometry_msgs::msg::Twist::SharedPtr m){ cmd_v_=m->linear.x; cmd_w_=m->angular.z; last_cmd_=now(); });
    cmd_in_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel_in", 10, [this](geometry_msgs::msg::Twist::SharedPtr m){ cmd_v_=m->linear.x; cmd_w_=m->angular.z; last_cmd_=now(); });

    phys_timer_ = create_wall_timer(20ms, std::bind(&PuzzlebotSim::stepPhysics, this));  // 50 Hz
    odom_timer_ = create_wall_timer(33ms, std::bind(&PuzzlebotSim::publishOdom, this));   // 30 Hz
    auto lidar_period = std::chrono::duration<double>(1.0 / lidar_hz_);
    lidar_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(lidar_period),
      std::bind(&PuzzlebotSim::publishScan, this));

    RCLCPP_INFO(get_logger(), "Puzzlebot sim (RPLidar A1M8): room %.2f×%.2f, "
                "%.1f Hz / %d beams (%.2f°), range %.2f–%.0f m, "
                "noise %.0f%%·d+%.0fmm, slip rot σ=%.0f%%",
                room_x_, room_y_, lidar_hz_, lidar_beams_, 360.0/lidar_beams_,
                range_min_, range_max_, range_noise_pct_*100, range_noise_floor_*1000,
                slip_rot_*100);
    RCLCPP_INFO(get_logger(), "  LiDAR latency = %.0f ms, stamp = %s",
                lidar_latency_ * 1000.0,
                stamp_on_publish_ ? "DELIVERY-time (BAD — models latent driver)"
                                  : "capture-time (correct)");
  }

private:
  void buildEnvironment() {
    double hx = room_x_/2, hy = room_y_/2;
    // outer walls
    segs_.push_back({-hx,-hy, hx,-hy});
    segs_.push_back({ hx,-hy, hx, hy});
    segs_.push_back({ hx, hy,-hx, hy});
    segs_.push_back({-hx, hy,-hx,-hy});
    // three interior racks (rectangles)
    auto rack = [&](double cx, double cy, double w, double h){
      segs_.push_back({cx-w/2,cy-h/2, cx+w/2,cy-h/2});
      segs_.push_back({cx+w/2,cy-h/2, cx+w/2,cy+h/2});
      segs_.push_back({cx+w/2,cy+h/2, cx-w/2,cy+h/2});
      segs_.push_back({cx-w/2,cy+h/2, cx-w/2,cy-h/2});
    };
    rack(-0.8, 0.6, 0.30, 1.0);
    rack( 0.5, 0.7, 1.0, 0.30);
    rack( 0.6,-0.6, 0.30, 1.0);
  }

  void publishStaticLaser() {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = now(); t.header.frame_id = "base_link"; t.child_frame_id = "laser";
    t.transform.translation.x = laser_x_;
    t.transform.rotation.z = std::sin(laser_yaw_/2);
    t.transform.rotation.w = std::cos(laser_yaw_/2);
    static_br_->sendTransform(t);
  }

  void stepPhysics() {
    double dt = 0.02;
    // command timeout → stop
    double v = cmd_v_, w = cmd_w_;
    if ((now() - last_cmd_).seconds() > 0.5) { v = 0; w = 0; }

    // ── TRUE motion: command + slip + noise ──
    std::normal_distribution<double> g(0.0, 1.0);
    double v_true = v * (1.0 + slip_trans_ * g(rng_));
    // rotation slips MORE: wheels spin but robot turns less, plus noise
    double w_true = w * (1.0 - std::abs(slip_rot_ * g(rng_)) * 0.5) + slip_rot_ * w * g(rng_) * 0.5;
    tth_ = std::atan2(std::sin(tth_ + w_true*dt), std::cos(tth_ + w_true*dt));
    tx_ += v_true * std::cos(tth_) * dt;
    ty_ += v_true * std::sin(tth_) * dt;

    // ── ODOM motion: what the encoders report (command + small enc noise,
    // but NO knowledge of the slip) → drifts away from ground truth ──
    double v_enc = v + odom_noise_v_ * g(rng_);
    double w_enc = w + odom_noise_w_ * g(rng_);
    odth_ = std::atan2(std::sin(odth_ + w_enc*dt), std::cos(odth_ + w_enc*dt));
    odx_ += v_enc * std::cos(odth_) * dt;
    ody_ += v_enc * std::sin(odth_) * dt;

    releaseScans();   // deliver any scans whose latency has elapsed
  }

  void publishOdom() {
    auto t = now();
    nav_msgs::msg::Odometry o;
    // child = base_footprint (ground frame), matching the real robot's
    // real_odom.py.  robot_state_publisher then adds the fixed
    // base_footprint->base_link (z=0.05) and base_link->{wheels,lidar,...} so
    // the full URDF model can be visualised without base_link getting two
    // parents (which would break the TF tree).
    o.header.stamp = t; o.header.frame_id = "odom"; o.child_frame_id = "base_footprint";
    o.pose.pose.position.x = odx_; o.pose.pose.position.y = ody_;
    o.pose.pose.orientation.z = std::sin(odth_/2); o.pose.pose.orientation.w = std::cos(odth_/2);
    o.twist.twist.linear.x = cmd_v_; o.twist.twist.angular.z = cmd_w_;
    odom_pub_->publish(o);

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = t; tf.header.frame_id = "odom"; tf.child_frame_id = "base_footprint";
    tf.transform.translation.x = odx_; tf.transform.translation.y = ody_;
    tf.transform.rotation.z = std::sin(odth_/2); tf.transform.rotation.w = std::cos(odth_/2);
    tf_br_->sendTransform(tf);

    geometry_msgs::msg::PoseStamped gt;
    gt.header.stamp = t; gt.header.frame_id = "map";
    gt.pose.position.x = tx_; gt.pose.position.y = ty_;
    gt.pose.orientation.z = std::sin(tth_/2); gt.pose.orientation.w = std::cos(tth_/2);
    gt_pub_->publish(gt);

    // accumulate + publish the two trajectories (throttled)
    if (++path_div_ % 5 == 0) {
      geometry_msgs::msg::PoseStamped op;
      op.header.stamp = t; op.header.frame_id = "map";
      op.pose.position.x = odx_; op.pose.position.y = ody_;
      op.pose.orientation.z = std::sin(odth_/2); op.pose.orientation.w = std::cos(odth_/2);
      odom_path_.poses.push_back(op);
      gt_path_.poses.push_back(gt);
      if (odom_path_.poses.size() > 2000) { odom_path_.poses.erase(odom_path_.poses.begin()); gt_path_.poses.erase(gt_path_.poses.begin()); }
      odom_path_.header.stamp = t; gt_path_.header.stamp = t;
      odom_path_pub_->publish(odom_path_);
      gt_path_pub_->publish(gt_path_);
    }
  }

  // ray vs all segments → nearest hit distance from (ox,oy) heading a
  double raycast(double ox, double oy, double a) {
    double dx = std::cos(a), dy = std::sin(a);
    double best = range_max_;
    for (const auto& s : segs_) {
      double x3=s.x1,y3=s.y1,x4=s.x2,y4=s.y2;
      double den = (x4-x3)*(-dy) - (y4-y3)*(-dx);
      if (std::abs(den) < 1e-12) continue;
      double t = ((ox-x3)*(-dy) - (oy-y3)*(-dx)) / den;     // along segment
      double u = ((x4-x3)*(oy-y3) - (y4-y3)*(ox-x3)) / den; // along ray
      if (t >= 0 && t <= 1 && u >= 0) best = std::min(best, u);
    }
    return best;
  }

  void publishScan() {
    auto t = now();
    // laser origin in world (ground truth)
    double c = std::cos(tth_), s = std::sin(tth_);
    double lox = tx_ + c*laser_x_;
    double loy = ty_ + s*laser_x_;
    double lbase = tth_ + laser_yaw_;

    sensor_msgs::msg::LaserScan ls;
    ls.header.frame_id = "laser";
    ls.angle_min = -M_PI; ls.angle_max = M_PI;
    ls.angle_increment = 2.0*M_PI / lidar_beams_;
    ls.range_min = range_min_; ls.range_max = range_max_;
    ls.scan_time = 1.0 / lidar_hz_;
    ls.time_increment = ls.scan_time / lidar_beams_;
    ls.ranges.resize(lidar_beams_);

    std::normal_distribution<double> g(0.0, 1.0);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < lidar_beams_; ++i) {
      double a = lbase + ls.angle_min + i * ls.angle_increment;
      double r = raycast(lox, loy, a);
      if (r >= range_max_ || u(rng_) < dropout_prob_) {
        ls.ranges[i] = std::numeric_limits<float>::infinity();
      } else {
        // A1 accuracy: σ = 1% of distance + small fixed floor.
        double sigma = range_noise_pct_ * r + range_noise_floor_;
        r += g(rng_) * sigma;
        ls.ranges[i] = static_cast<float>(std::max(range_min_, r));
      }
    }
    // capture-time stamp; deliver after latency
    double cap = t.seconds();
    if (lidar_latency_ <= 0.0) {
      ls.header.stamp = t;
      scan_pub_->publish(ls);
    } else {
      pending_.push_back({cap, ls});
    }
  }

  void releaseScans() {
    double tn = now().seconds();
    while (!pending_.empty() && tn - pending_.front().cap >= lidar_latency_) {
      auto pkt = pending_.front(); pending_.pop_front();
      // stamp: capture time (correct driver) OR delivery time (latent driver)
      double stamp = stamp_on_publish_ ? tn : pkt.cap;
      pkt.scan.header.stamp = rclcpp::Time(static_cast<int64_t>(stamp * 1e9), RCL_ROS_TIME);
      scan_pub_->publish(pkt.scan);
    }
  }

  std::vector<Seg> segs_;
  double room_x_, room_y_, range_max_, range_min_;
  double range_noise_pct_, range_noise_floor_, dropout_prob_;
  double slip_rot_, slip_trans_, odom_noise_v_, odom_noise_w_;
  double laser_x_, laser_yaw_;
  double lidar_latency_; bool stamp_on_publish_;
  struct Pkt { double cap; sensor_msgs::msg::LaserScan scan; };
  std::deque<Pkt> pending_;
  int lidar_beams_; double lidar_hz_;
  double tx_, ty_, tth_, odx_, ody_, odth_;
  double cmd_v_=0, cmd_w_=0;
  rclcpp::Time last_cmd_{0,0,RCL_ROS_TIME};
  std::mt19937 rng_;

  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr gt_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr odom_path_pub_, gt_path_pub_;
  nav_msgs::msg::Path odom_path_, gt_path_;
  int path_div_ = 0;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_, cmd_in_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_br_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_br_;
  rclcpp::TimerBase::SharedPtr phys_timer_, odom_timer_, lidar_timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PuzzlebotSim>());
  rclcpp::shutdown();
  return 0;
}
