// slam_node.cpp — pure C++ Puzzlebot SLAM node.
//
// Pipeline per scan:
//   deskew → scan-to-scan ICP odom → AMCL predict/update/resample
//   → scan-to-map ICP refine → motion+certainty+fit-gated integration
//   → publish /slam_pose, /particle_cloud, /map, TF map→odom.

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <cstdlib>

// Best-effort per-thread priority lowering (Linux). Keeps the back-end
// optimiser from stealing CPU from the real-time front-end on the Nano.
#if defined(__linux__)
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "slam/se2.hpp"
#include "slam/occupancy_grid.hpp"
#include "slam/amcl.hpp"
#include "slam/icp.hpp"
#include "slam/pose_graph.hpp"

using namespace std::chrono_literals;
using slam::Pose2; using slam::wrap; using slam::compose; using slam::inverse; using slam::relative;

static double yawFromQuat(const geometry_msgs::msg::Quaternion& q) {
  double siny = 2.0 * (q.w * q.z + q.x * q.y);
  double cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny, cosy);
}

// Expand a leading '~' to $HOME so map_yaml params like '~/ros2_maps/...'
// resolve the same way the Python launch files (os.path.expanduser) do.
static std::string expandUser(const std::string& p) {
  if (!p.empty() && p[0] == '~') {
    const char* home = std::getenv("HOME");
    if (home) return std::string(home) + p.substr(1);
  }
  return p;
}

class SlamNode : public rclcpp::Node {
public:
  SlamNode() : Node("slam_node") {
    // ── params ──
    res_   = declare_parameter("resolution", 0.04);
    mapw_  = declare_parameter("map_width", 400);
    maph_  = declare_parameter("map_height", 400);
    double l_occ = declare_parameter("l_occ", 0.5);
    double l_free = declare_parameter("l_free", -0.2);
    double l_min = declare_parameter("l_min", -2.0);
    double l_max = declare_parameter("l_max", 4.0);
    double dl_occ = declare_parameter("display_l_occ", 2.5);
    double dl_free = declare_parameter("display_l_free", -1.2);
    double occ_stop = declare_parameter("occupied_stop", 3.0);
    range_min_ = declare_parameter("range_min", 0.12);
    range_max_ = declare_parameter("range_max", 10.0);
    ray_step_  = declare_parameter("ray_step", 1);
    // RPLIDAR A1 reports a per-sample quality in LaserScan.intensities.
    // Drop weak returns (specular/oblique hits, glass, range-edge noise)
    // before they reach the matcher.  0.0 disables (e.g. the sim, which
    // publishes no intensities); the real A1 driver scales quality ~0–47.
    quality_min_ = declare_parameter("scan_quality_min", 0.0);
    outlier_max_jump_ = declare_parameter("outlier_max_jump", 0.3);
    map_pub_every_ = declare_parameter("map_publish_every", 5);
    base_frame_ = declare_parameter("base_frame", std::string("base_link"));
    min_dxy_ = declare_parameter("min_delta_xy", 0.02);
    min_dth_ = declare_parameter("min_delta_theta", 0.02);
    conf_thresh_ = declare_parameter("amcl_confidence_threshold", 0.95);
    conf_rxy_ = declare_parameter("amcl_confident_radius_xy", 0.15);
    conf_rth_ = declare_parameter("amcl_confident_radius_theta", 0.10);
    // Subtracted from the scan's header.stamp before looking up odom.
    // Compensates a LiDAR driver that timestamps at delivery instead of
    // capture (or any known sensor latency).  Set to the measured
    // capture-to-delivery delay (e.g. 0.15 for a 150 ms latent A1).
    scan_time_offset_ = declare_parameter("scan_time_offset", 0.0);
    sm_odom_en_ = declare_parameter("scan_match_odom_enabled", true);
    sm_max_fit_ = declare_parameter("scan_match_max_fitness", 0.08);
    sm_max_jump_ = declare_parameter("scan_match_max_jump", 0.50);
    sm_reject_ = declare_parameter("scan_match_reject", 0.40);
    s2m_en_ = declare_parameter("scan_to_map_refine_enabled", true);
    s2m_radius_ = declare_parameter("scan_to_map_radius", 4.0);
    s2m_minpts_ = declare_parameter("scan_to_map_min_points", 80);
    s2m_reject_ = declare_parameter("scan_to_map_reject", 0.30);
    s2m_maxcorr_ = declare_parameter("scan_to_map_max_correction", 0.15);
    s2m_maxcorr_deg_ = declare_parameter("scan_to_map_max_correction_deg", 8.0);
    // Coarse-to-fine yaw search (the corner safety net): before the fine
    // point-to-line ICP, sweep a few discrete heading hypotheses about the
    // robot centre and keep the best.  Without an IMU this is what rescues
    // a turn where wheel-odom yaw drifted past the fine ICP's small
    // correction gate.  Set range to 0 to disable.
    s2m_coarse_range_deg_ = declare_parameter("scan_to_map_coarse_range_deg", 6.0);
    s2m_coarse_step_deg_  = declare_parameter("scan_to_map_coarse_step_deg", 1.5);
    integ_max_fit_ = declare_parameter("integrate_max_map_fitness", 0.10);
    bootstrap_cells_ = declare_parameter("bootstrap_min_cells", 2500);
    // ── Graph SLAM / loop closure ──
    graph_en_ = declare_parameter("graph_enabled", true);
    kf_dist_ = declare_parameter("graph_keyframe_dist", 0.25);
    kf_angle_ = declare_parameter("graph_keyframe_angle", 0.25);
    loop_radius_ = declare_parameter("graph_loop_radius", 0.45);
    loop_exclude_ = declare_parameter("graph_loop_exclude_recent", 20);
    loop_min_fit_ = declare_parameter("graph_loop_min_fitness", 0.06);
    graph_opt_every_ = declare_parameter("graph_optimize_every_kf", 5);
    graph_huber_ = declare_parameter("graph_huber", 0.5);
    init_x_ = declare_parameter("initial_x", 0.0);
    init_y_ = declare_parameter("initial_y", 0.0);
    init_th_ = declare_parameter("initial_theta", 0.0);
    init_sxy_ = declare_parameter("initial_sigma_xy", 0.05);
    init_sth_ = declare_parameter("initial_sigma_theta", 0.02);
    tf_alpha_ = declare_parameter("tf_smoothing_alpha", 0.7);
    // Heavy RViz-only publishers (particle cloud + graph markers).  Default on
    // for the laptop/dev box; set false on the 2 GB Jetson Nano to free CPU +
    // WiFi bandwidth (the scan front-end never needs them).
    publish_debug_ = declare_parameter("publish_debug", true);
    // Per-scan processing-time budget (ms).  At 10 Hz a scan must finish in
    // 100 ms; warn (throttled) past this so real-hardware overruns surface.
    scan_budget_ms_ = declare_parameter("scan_budget_ms", 80.0);
    // ── Saved-map preload (localisation-only) ────────────────────────────
    // start_mode=navigation|localization + a readable map_yaml → load the
    // saved grid and localise against it WITHOUT ever rebuilding it.  This is
    // the "I already mapped, now just navigate" flow.  start_mode=mapping
    // (or an empty/unreadable map_yaml) → live mapping from a fresh grid.
    std::string map_yaml   = declare_parameter("map_yaml", std::string(""));
    std::string start_mode = declare_parameter("start_mode", std::string("mapping"));

    slam::AmclParams ap;
    ap.n_particles = declare_parameter("amcl_n_particles", 600);
    ap.alpha1 = declare_parameter("amcl_alpha1", 0.10);
    ap.alpha2 = declare_parameter("amcl_alpha2", 0.08);
    ap.alpha3 = declare_parameter("amcl_alpha3", 0.10);
    ap.alpha4 = declare_parameter("amcl_alpha4", 0.05);
    ap.sigma_hit = declare_parameter("amcl_sigma_hit", 0.10);
    ap.z_hit = declare_parameter("amcl_z_hit", 0.85);
    ap.z_rand = declare_parameter("amcl_z_rand", 0.15);
    ap.z_max_dist = declare_parameter("amcl_z_max_dist", 2.0);
    ap.max_scan_pts = declare_parameter("amcl_max_scan_pts", 240);
    ap.neff_ratio = declare_parameter("amcl_neff_threshold_ratio", 0.5);
    ap.recovery_neff_ratio = declare_parameter("amcl_recovery_neff_ratio", 0.15);
    ap.recovery_sigma_xy = declare_parameter("amcl_recovery_sigma_xy", 0.10);
    ap.recovery_sigma_theta = declare_parameter("amcl_recovery_sigma_theta", 0.08);
    ap.use_cuda = declare_parameter("amcl_use_cuda", true);

    grid_ = std::make_unique<slam::OccupancyGrid>(mapw_, maph_, res_, l_occ, l_free,
                                                  l_min, l_max, dl_occ, dl_free, occ_stop);
    amcl_ = std::make_unique<slam::AMCL>(grid_.get(), ap);

    sx_ = init_x_; sy_ = init_y_; sth_ = init_th_;
    tf_x_ = sx_; tf_y_ = sy_; tf_th_ = sth_;
    amcl_->initGaussian(sx_, sy_, sth_, init_sxy_, init_sth_);
    RCLCPP_INFO(get_logger(), "AMCL initialised at (%.2f, %.2f, %.1f°) %s",
                sx_, sy_, sth_ * 180.0 / M_PI,
                amcl_->cudaActive() ? "[CUDA]" : "[CPU]");

    // ── Optional: preload a saved map → LOCALISATION-ONLY ────────────────
    // In navigation/localization mode, load the saved grid and localise
    // against it without ever modifying it: the map-write path (maybeIntegrate)
    // AND the graph back-end's map re-rasterise are both disabled, so a good
    // saved map is never wiped by a fresh one.  If the file can't be read
    // (e.g. first run, no map yet) we warn and fall back to live mapping so a
    // fresh robot still builds a map.
    const bool want_localize = (start_mode == "navigation" || start_mode == "localization");
    if (want_localize && !map_yaml.empty()) {
      std::string err, path = expandUser(map_yaml);
      if (grid_->loadFromYaml(path, err)) {
        localization_only_ = true;
        map_path_       = path;
        graph_en_       = false;   // no pose-graph/loop-closure map rebuild
        bootstrap_done_ = true;    // map already established
        first_write_    = false;
        lmx_ = sx_; lmy_ = sy_; lmth_ = sth_;
        RCLCPP_INFO(get_logger(),
          "Loaded saved map '%s' (%d wall cells) — LOCALISATION-ONLY: the map "
          "will NOT be modified. If the robot doesn't start at (%.2f, %.2f, %.0f°), "
          "set its pose with RViz '2D Pose Estimate'.",
          path.c_str(), grid_->countOccupied(), init_x_, init_y_, init_th_ * 180.0 / M_PI);
      } else {
        RCLCPP_WARN(get_logger(),
          "start_mode=%s but could not load map '%s' (%s) — falling back to "
          "LIVE MAPPING from an empty grid.",
          start_mode.c_str(), map_yaml.c_str(), err.c_str());
      }
    } else if (!want_localize) {
      RCLCPP_INFO(get_logger(), "start_mode=%s — LIVE MAPPING (fresh grid).",
                  start_mode.c_str());
    }

    // ── interfaces ──
    auto qos_scan = rclcpp::SensorDataQoS();
    // /map is latched (transient_local): a preloaded map and the live grid
    // both survive for late joiners — nav/RViz/map_saver on the laptop get
    // the last map immediately on connect instead of waiting for the next
    // periodic republish (which matters most across WiFi).
    map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
        "/map", rclcpp::QoS(1).transient_local());
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/slam_pose", 10);
    parts_pub_ = create_publisher<geometry_msgs::msg::PoseArray>("/particle_cloud", 1);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("/slam_path", 1);
    graph_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/graph_markers", 1);
    tf_br_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", 10, std::bind(&SlamNode::onOdom, this, std::placeholders::_1));
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", qos_scan, std::bind(&SlamNode::onScan, this, std::placeholders::_1));
    initpose_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/initialpose", 10, std::bind(&SlamNode::onInitPose, this, std::placeholders::_1));

    reset_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/reset_map", std::bind(&SlamNode::onReset, this,
                               std::placeholders::_1, std::placeholders::_2));

    tf_timer_ = create_wall_timer(33ms, [this]() { publishTf(now()); });

    // ── Graph back-end runs on its own low-priority thread (§5.3): LM
    // optimisation + map re-rasterisation never block the scan front-end. ──
    if (graph_en_)
      backend_thread_ = std::thread(&SlamNode::backendLoop, this);

    // Jetson Nano 2 GB headroom check: the log grid + likelihood field + ROS
    // overhead leave little margin, and the back-end clone spikes it further.
    // Warn loudly (don't abort — a healthy edge case may still run) so a
    // memory-starved boot is diagnosable instead of a silent OOM kill.
    long avail_kb = readMemAvailableKb();
    if (avail_kb > 0) {
      RCLCPP_INFO(get_logger(), "MemAvailable at startup: %ld MB", avail_kb / 1024);
      if (avail_kb < 600 * 1024)
        RCLCPP_WARN(get_logger(),
          "Low memory (%ld MB < 600 MB) — on a 2 GB Jetson set publish_debug:=false, "
          "do NOT run Gazebo on the robot, and watch for OOM during loop closure.",
          avail_kb / 1024);
    }

    RCLCPP_INFO(get_logger(), "SLAM (C++) started — %dx%d @ %.3f m/cell  (debug pubs: %s)",
                mapw_, maph_, res_, publish_debug_ ? "on" : "off");

    // Publish the preloaded map once so latched late-joiners (nav/RViz on the
    // laptop) see it immediately, before the first scan triggers a republish.
    if (localization_only_) publishMap(now());
  }

  ~SlamNode() override {
    { std::lock_guard<std::mutex> lk(be_mtx_); backend_stop_ = true; }
    be_cv_.notify_all();
    if (backend_thread_.joinable()) backend_thread_.join();
  }

private:
  // ── odom ──
  void onOdom(const nav_msgs::msg::Odometry::SharedPtr m) {
    std::lock_guard<std::mutex> lk(mtx_);
    ox_ = m->pose.pose.position.x;
    oy_ = m->pose.pose.position.y;
    oth_ = yawFromQuat(m->pose.pose.orientation);
    odom_ready_ = true;
    double t = m->header.stamp.sec + m->header.stamp.nanosec * 1e-9;
    odom_buf_.push_back({t, ox_, oy_, oth_});
    if (odom_buf_.size() > 200) odom_buf_.pop_front();
  }

  void odomAt(double t, double& x, double& y, double& th) {
    if (odom_buf_.empty()) { x = ox_; y = oy_; th = oth_; return; }
    if (t <= odom_buf_.front().t) { x = odom_buf_.front().x; y = odom_buf_.front().y; th = odom_buf_.front().th; return; }
    if (t >= odom_buf_.back().t)  { x = odom_buf_.back().x;  y = odom_buf_.back().y;  th = odom_buf_.back().th; return; }
    for (size_t i = 1; i < odom_buf_.size(); ++i) {
      if (odom_buf_[i].t >= t) {
        auto& a = odom_buf_[i-1]; auto& b = odom_buf_[i];
        double f = (b.t > a.t) ? (t - a.t) / (b.t - a.t) : 0.0;
        x = a.x + f * (b.x - a.x);
        y = a.y + f * (b.y - a.y);
        th = wrap(a.th + f * wrap(b.th - a.th));
        return;
      }
    }
    x = odom_buf_.back().x; y = odom_buf_.back().y; th = odom_buf_.back().th;
  }

  // /proc/meminfo MemAvailable in kB (-1 if unreadable, e.g. non-Linux).
  static long readMemAvailableKb() {
    std::ifstream f("/proc/meminfo");
    std::string key; long val; std::string unit;
    while (f >> key >> val >> unit) {
      if (key == "MemAvailable:") return val;
    }
    return -1;
  }

  bool resolveLaser(const std::string& frame) {
    if (laser_ok_) return true;
    try {
      auto tf = tf_buffer_->lookupTransform(base_frame_, frame, tf2::TimePointZero);
      lx_ = tf.transform.translation.x;
      ly_ = tf.transform.translation.y;
      lyaw_ = yawFromQuat(tf.transform.rotation);
      laser_ok_ = true;
      RCLCPP_INFO(get_logger(), "Laser→%s: x=%.3f y=%.3f yaw=%.1f°",
                  base_frame_.c_str(), lx_, ly_, lyaw_ * 180.0 / M_PI);
    } catch (...) { return false; }
    return true;
  }

  // build deskewed base-frame cloud
  void buildCloud(const sensor_msgs::msg::LaserScan::SharedPtr m,
                  double scan_t, double scan_dur,
                  std::vector<float>& cx, std::vector<float>& cy) {
    const int N = static_cast<int>(m->ranges.size());
    // A1 quality gate: only active when the driver fills intensities
    // 1:1 with ranges and a positive threshold is configured.
    const bool use_quality = quality_min_ > 0.0 &&
                             m->intensities.size() == m->ranges.size();
    // scan-start odom
    double sx0, sy0, sth0; odomAt(scan_t, sx0, sy0, sth0);
    double cL = std::cos(lyaw_), sL = std::sin(lyaw_);
    cx.clear(); cy.clear(); cx.reserve(N); cy.reserve(N);
    for (int i = 0; i < N; i += ray_step_) {
      double r = m->ranges[i];
      if (!std::isfinite(r) || r < range_min_ || r > range_max_) continue;
      if (use_quality && m->intensities[i] < quality_min_) continue;
      // simple outlier: differs from both neighbours by > jump
      if (i > 0 && i < N-1) {
        double rp = m->ranges[i-1], rn = m->ranges[i+1];
        if (std::isfinite(rp) && std::isfinite(rn) &&
            std::abs(r-rp) > outlier_max_jump_ && std::abs(r-rn) > outlier_max_jump_)
          continue;
      }
      // ray capture time + deskew delta
      double frac = (N > 1) ? static_cast<double>(i) / (N - 1) : 0.5;
      double qt = scan_t + frac * scan_dur;
      double qx, qy, qth; odomAt(qt, qx, qy, qth);
      double dxw = qx - sx0, dyw = qy - sy0, dth = wrap(qth - sth0);
      double cc = std::cos(sth0), ss = std::sin(sth0);
      double dxb = cc * dxw + ss * dyw;
      double dyb = -ss * dxw + cc * dyw;
      // laser-frame hit
      double a = m->angle_min + i * m->angle_increment;
      double rxl = r * std::cos(a), ryl = r * std::sin(a);
      double bxc = lx_ + cL * rxl - sL * ryl;
      double byc = ly_ + sL * rxl + cL * ryl;
      double cd = std::cos(dth), sd = std::sin(dth);
      cx.push_back(static_cast<float>(dxb + cd * bxc - sd * byc));
      cy.push_back(static_cast<float>(dyb + sd * bxc + cd * byc));
    }
  }

  void onScan(const sensor_msgs::msg::LaserScan::SharedPtr m) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!odom_ready_) return;
    if (!resolveLaser(m->header.frame_id)) return;
    const auto proc_t0 = std::chrono::steady_clock::now();
    double scan_t = m->header.stamp.sec + m->header.stamp.nanosec * 1e-9
                    - scan_time_offset_;   // compensate sensor latency
    double scan_dur = (m->scan_time > 0.0) ? m->scan_time : 0.1;
    double scan_ox, scan_oy, scan_oth; odomAt(scan_t, scan_ox, scan_oy, scan_oth);

    std::vector<float> cx, cy;
    buildCloud(m, scan_t, scan_dur, cx, cy);
    if (cx.size() < 20) return;

    // ── scan-to-scan ICP odom → AMCL predict delta ──
    double dxb, dyb, dth;
    {
      double dxw = ox_ - po_x_, dyw = oy_ - po_y_, dthw = wrap(oth_ - po_th_);
      double cc = std::cos(po_th_), ss = std::sin(po_th_);
      dxb = cc * dxw + ss * dyw; dyb = -ss * dxw + cc * dyw; dth = dthw;
    }
    if (!odom_init_) { odom_init_ = true; po_x_ = ox_; po_y_ = oy_; po_th_ = oth_; prev_cloud_x_ = cx; prev_cloud_y_ = cy; return; }
    if (sm_odom_en_ && !prev_cloud_x_.empty()) {
      slam::NNIndex idx(prev_cloud_x_, prev_cloud_y_, sm_reject_);
      auto r = slam::icpMatch(cx, cy, idx, dxb, dyb, dth, 20, sm_reject_, 20);
      if (r.ok && r.fitness < sm_max_fit_ &&
          std::hypot(r.dx, r.dy) < sm_max_jump_ && std::abs(r.dth) < M_PI/4)
        { dxb = r.dx; dyb = r.dy; dth = r.dth; }
    }
    prev_cloud_x_ = cx; prev_cloud_y_ = cy;
    po_x_ = ox_; po_y_ = oy_; po_th_ = oth_;

    amcl_->predict(dxb, dyb, dth);
    amcl_->update(cx, cy);
    amcl_->resample();
    Pose2 e = amcl_->estimate();
    sx_ = e.x; sy_ = e.y; sth_ = e.th;

    // ── scan-to-map refine + fit gate ──
    double map_fit = refineToMap(cx, cy);

    // map→odom (smoothed)
    Pose2 newtf = compose({sx_, sy_, sth_}, inverse({ox_, oy_, oth_}));
    if (tf_alpha_ < 1.0) {
      tf_x_ = tf_alpha_ * newtf.x + (1 - tf_alpha_) * tf_x_;
      tf_y_ = tf_alpha_ * newtf.y + (1 - tf_alpha_) * tf_y_;
      tf_th_ = wrap(tf_th_ + tf_alpha_ * wrap(newtf.th - tf_th_));
    } else { tf_x_ = newtf.x; tf_y_ = newtf.y; tf_th_ = newtf.th; }

    // integrate at scan-start pose
    Pose2 scan_slam = compose({tf_x_, tf_y_, tf_th_}, {scan_ox, scan_oy, scan_oth});
    maybeIntegrate(scan_slam, cx, cy, map_fit);

    // ── Graph SLAM: keyframes + loop closure ──
    bool new_kf = graph_en_ && graphStep(cx, cy);

    auto t = now();
    publishPose(t);
    publishTf(t);
    if (++scan_count_ % map_pub_every_ == 0) publishMap(t);
    // Path/graph markers only change on a keyframe — publish them then, not
    // on every scan (they'd otherwise ship the whole graph over WiFi at scan
    // rate for RViz).
    if (new_kf) { publishPath(t); if (publish_debug_) publishGraph(t); }

    // Real-time watchdog: at 10 Hz the whole per-scan pipeline must fit 100 ms.
    // Warn (throttled) past the budget so a thermally-throttled / on-battery
    // Nano overrun is visible instead of silently dropping scans.
    const double proc_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - proc_t0).count();
    if (proc_ms > scan_budget_ms_)
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "Scan processing %.1f ms > %.0f ms budget — front-end may not sustain 10 Hz.",
        proc_ms, scan_budget_ms_);
  }

  // Keyframe selection, loop detection, periodic optimisation.
  // Returns true iff a new keyframe was added this scan.
  bool graphStep(const std::vector<float>& cx, const std::vector<float>& cy) {
    bool first = (graph_.size() == 0);
    if (!first) {
      double d = std::hypot(sx_ - kf_x_, sy_ - kf_y_);
      double a = std::abs(wrap(sth_ - kf_th_));
      if (d < kf_dist_ && a < kf_angle_) return false;   // not enough motion
    }
    int nid = graph_.addNode({sx_, sy_, sth_}, cx, cy);
    if (!first) {
      Pose2 meas = relative({kf_x_, kf_y_, kf_th_}, {sx_, sy_, sth_});
      graph_.addOdomEdge(last_kf_, nid, meas);
    }
    kf_x_ = sx_; kf_y_ = sy_; kf_th_ = sth_; last_kf_ = nid;

    // loop closure: match against older nearby keyframes
    auto cands = graph_.neighbours(nid, loop_radius_, loop_exclude_);
    for (int cid : cands) {
      const auto& old = graph_.nodes()[cid];
      // ICP current scan onto the old keyframe scan, seeded by their
      // current relative-pose estimate.
      Pose2 init = relative(old.pose, {sx_, sy_, sth_});
      slam::NNIndex idx(old.scan_x, old.scan_y, loop_min_fit_ * 2 + 0.1);
      auto r = slam::icpMatch(cx, cy, idx, init.x, init.y, init.th, 25, 0.3, 20);
      if (r.ok && r.fitness < loop_min_fit_) {
        graph_.addLoopEdge(cid, nid, {r.dx, r.dy, r.dth}, 0.03, 0.015);
        RCLCPP_INFO(get_logger(), "Loop closure: kf %d ↔ %d (fit %.1f cm, total %d)",
                    nid, cid, r.fitness * 100, graph_.loopCount());
        break;
      }
    }

    // Wake the back-end thread to optimise (and re-rasterise the map) when
    // we have loops.  We deliberately do NOT hard-snap the live AMCL pose —
    // that jump smears the live grid.  The front-end (AMCL + scan-match +
    // scan-to-map) keeps tracking; the back-end is the consistency layer
    // that corrects the keyframe trajectory and rebuilds a clean map (§5.4).
    if (graph_.loopCount() > 0 && (graph_.size() % graph_opt_every_ == 0)) {
      { std::lock_guard<std::mutex> lk(be_mtx_); backend_wanted_ = true; }
      be_cv_.notify_one();
    }
    return true;
  }

  // ── Back-end worker (§5.3, §5.4) ────────────────────────────────────
  // Snapshots the graph, runs LM + Huber off-thread, re-rasterises the
  // grid from the corrected keyframe scans, then commits both atomically.
  void backendLoop() {
#if defined(__linux__)
    // Nice the back-end down so the A57 cores favour the front-end.
    setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), 10);
#endif
    std::unique_lock<std::mutex> lk(be_mtx_);
    while (true) {
      be_cv_.wait(lk, [this] { return backend_wanted_ || backend_stop_; });
      if (backend_stop_) break;
      backend_wanted_ = false;
      lk.unlock();

      // 1. snapshot graph + laser offset under the main lock
      slam::PoseGraph snap;
      double slx, sly;
      {
        std::lock_guard<std::mutex> mlk(mtx_);
        snap = graph_;
        slx = lx_; sly = ly_;
      }

      // 2. optimise the snapshot (heavy, no lock held)
      double ie = 0, fe = 0;
      bool improved = (snap.loopCount() > 0) &&
                      snap.optimize(30, graph_huber_, ie, fe);

      if (improved) {
        // 3. re-rasterise a fresh grid from corrected keyframe scans
        auto rebuilt = grid_->cloneEmpty();   // geometry config is immutable
        for (const auto& nd : snap.nodes()) {
          slam::Cloud ep; ep.reserve(nd.scan_x.size());
          double c = std::cos(nd.pose.th), s = std::sin(nd.pose.th);
          for (size_t i = 0; i < nd.scan_x.size(); ++i)
            ep.push(static_cast<float>(c*nd.scan_x[i]-s*nd.scan_y[i]+nd.pose.x),
                    static_cast<float>(s*nd.scan_x[i]+c*nd.scan_y[i]+nd.pose.y));
          rebuilt->integrateCloud(nd.pose, ep, slx, sly);
        }

        // 4. commit: corrected poses + clean map, atomically
        {
          std::lock_guard<std::mutex> mlk(mtx_);
          int n = std::min(snap.size(), graph_.size());
          for (int i = 0; i < n; ++i) graph_.node(i).pose = snap.nodes()[i].pose;
          grid_->adoptLogFrom(*rebuilt);
        }
        RCLCPP_INFO(get_logger(),
          "Backend: %d nodes / %d loops optimised (err %.1f→%.1f) + map re-rasterised",
          snap.size(), snap.loopCount(), ie, fe);
      }

      lk.lock();
    }
  }

  // Mean clamped nearest-neighbour distance of the base cloud projected to
  // world at heading `theta` about (sx_,sy_) — the coarse-search score.
  double coarseScore(const std::vector<float>& cx, const std::vector<float>& cy,
                     const slam::NNIndex& idx, double theta) const {
    double c = std::cos(theta), s = std::sin(theta);
    double acc = 0.0; int n = 0;
    const int stride = std::max<int>(1, static_cast<int>(cx.size()) / 120);
    for (size_t i = 0; i < cx.size(); i += stride) {
      double wx = c * cx[i] - s * cy[i] + sx_;
      double wy = s * cx[i] + c * cy[i] + sy_;
      double d2; idx.nearest(wx, wy, d2);
      acc += std::min(std::sqrt(d2), s2m_reject_);
      ++n;
    }
    return n ? acc / n : std::numeric_limits<double>::infinity();
  }

  double refineToMap(const std::vector<float>& cx, const std::vector<float>& cy) {
    if (!s2m_en_) return 0.0;
    std::vector<float> mx, my;
    grid_->occupiedPoints(sx_, sy_, s2m_radius_, mx, my);
    if (static_cast<int>(mx.size()) < s2m_minpts_) return std::numeric_limits<double>::infinity();
    slam::NNIndex idx(mx, my, s2m_reject_);

    // ── Level 1: coarse yaw sweep about the robot centre ──
    if (s2m_coarse_range_deg_ > 0.0 && s2m_coarse_step_deg_ > 0.0) {
      double rng = s2m_coarse_range_deg_ * M_PI / 180.0;
      double stp = s2m_coarse_step_deg_  * M_PI / 180.0;
      double base = coarseScore(cx, cy, idx, sth_);
      double best_dth = 0.0, best = base;
      for (double d = -rng; d <= rng + 1e-9; d += stp) {
        if (std::abs(d) < 1e-9) continue;
        double sc = coarseScore(cx, cy, idx, sth_ + d);
        if (sc < best) { best = sc; best_dth = d; }
      }
      // Only adopt a non-trivial rotation if it clearly beats staying put.
      if (best_dth != 0.0 && best < base * 0.95)
        sth_ = wrap(sth_ + best_dth);
    }

    // ── Level 2: fine point-to-line ICP from the (now coarse-aligned) pose ──
    std::vector<float> wx(cx.size()), wy(cy.size());
    double c = std::cos(sth_), s = std::sin(sth_);
    for (size_t i = 0; i < cx.size(); ++i) {
      wx[i] = static_cast<float>(c * cx[i] - s * cy[i] + sx_);
      wy[i] = static_cast<float>(s * cx[i] + c * cy[i] + sy_);
    }
    auto r = slam::icpMatch(wx, wy, idx, 0, 0, 0, 20, s2m_reject_, 20);
    if (!r.ok) return std::numeric_limits<double>::infinity();
    double maxth = s2m_maxcorr_deg_ * M_PI / 180.0;
    if (std::hypot(r.dx, r.dy) <= s2m_maxcorr_ && std::abs(r.dth) <= maxth) {
      Pose2 nw = compose({r.dx, r.dy, r.dth}, {sx_, sy_, sth_});
      sx_ = nw.x; sy_ = nw.y; sth_ = nw.th;
    }
    return r.fitness;
  }

  void maybeIntegrate(const Pose2& sp, const std::vector<float>& cx,
                      const std::vector<float>& cy, double map_fit) {
    if (localization_only_) return;   // never modify a preloaded saved map
    bool moved = first_write_ ||
      std::hypot(sp.x - lmx_, sp.y - lmy_) >= min_dxy_ ||
      std::abs(wrap(sp.th - lmth_)) >= min_dth_;
    if (!moved) return;
    // Bootstrap phase: while the map is still sparse, integrate liberally
    // (skip the certainty gate) so the room outline can build up.  AMCL
    // can't reach high certainty against a one-scan map — it needs the
    // outline first.  The scan-to-map fit gate (below) is inactive here
    // too (returns inf until there's enough map), so the bootstrap is
    // free to grow.  Once the map is established the strict gates engage.
    // Bootstrap (sparse-map) phase ends once for good — latch it so we stop
    // scanning the whole grid with countOccupied() on every scan thereafter.
    bool bootstrap = false;
    if (!bootstrap_done_) {
      if (grid_->countOccupied() < bootstrap_cells_) bootstrap = true;
      else bootstrap_done_ = true;
    }
    if (conf_thresh_ > 0.0 && !first_write_ && !bootstrap) {
      double cert = amcl_->certainty(conf_rxy_, conf_rth_);
      if (cert < conf_thresh_) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
          "Map write skipped — certainty %.0f%% < %.0f%%", cert*100, conf_thresh_*100);
        return;
      }
    }
    if (integ_max_fit_ > 0.0 && !first_write_ && std::isfinite(map_fit) && map_fit > integ_max_fit_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
        "Map write skipped — scan-to-map fit %.1f cm > %.0f cm", map_fit*100, integ_max_fit_*100);
      return;
    }
    // world endpoints at scan-slam pose
    slam::Cloud ep; ep.reserve(cx.size());
    double c = std::cos(sp.th), s = std::sin(sp.th);
    for (size_t i = 0; i < cx.size(); ++i)
      ep.push(static_cast<float>(c*cx[i]-s*cy[i]+sp.x), static_cast<float>(s*cx[i]+c*cy[i]+sp.y));
    grid_->integrateCloud(sp, ep, lx_, ly_);
    first_write_ = false; lmx_ = sp.x; lmy_ = sp.y; lmth_ = sp.th;
  }

  void onInitPose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr m) {
    std::lock_guard<std::mutex> lk(mtx_);
    sx_ = m->pose.pose.position.x; sy_ = m->pose.pose.position.y;
    sth_ = yawFromQuat(m->pose.pose.orientation);
    Pose2 t = compose({sx_, sy_, sth_}, inverse({ox_, oy_, oth_}));
    tf_x_ = t.x; tf_y_ = t.y; tf_th_ = t.th;
    amcl_->initGaussian(sx_, sy_, sth_, 0.20, 0.10);
    RCLCPP_INFO(get_logger(), "Pose set to (%.2f, %.2f, %.1f°)", sx_, sy_, sth_*180/M_PI);
  }

  void onReset(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
               std::shared_ptr<std_srvs::srv::Trigger::Response> resp) {
    std::lock_guard<std::mutex> lk(mtx_);
    // In localisation-only mode a "reset" must NOT wipe the saved map —
    // reload it from disk instead so the known walls survive.
    if (localization_only_) {
      std::string err;
      if (grid_->loadFromYaml(map_path_, err)) {
        amcl_->initGaussian(sx_, sy_, sth_, init_sxy_, init_sth_);
        resp->success = true; resp->message = "saved map reloaded, AMCL re-seeded";
        RCLCPP_INFO(get_logger(), "Saved map reloaded (localisation mode), AMCL re-seeded.");
      } else {
        resp->success = false; resp->message = "map reload failed: " + err;
        RCLCPP_WARN(get_logger(), "Reset ignored — map reload failed: %s", err.c_str());
      }
      return;
    }
    grid_->reset();
    first_write_ = true;
    bootstrap_done_ = false;
    amcl_->initGaussian(sx_, sy_, sth_, init_sxy_, init_sth_);
    resp->success = true; resp->message = "grid cleared";
    RCLCPP_INFO(get_logger(), "Grid reset, AMCL re-seeded.");
  }

  // ── publishers ──
  void publishPose(const rclcpp::Time& t) {
    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp = t; ps.header.frame_id = "map";
    ps.pose.position.x = sx_; ps.pose.position.y = sy_;
    ps.pose.orientation.z = std::sin(sth_/2); ps.pose.orientation.w = std::cos(sth_/2);
    pose_pub_->publish(ps);

    // /particle_cloud is RViz-only — skip it when debug pubs are off (Jetson).
    if (!publish_debug_) return;
    geometry_msgs::msg::PoseArray pa;
    pa.header = ps.header;
    int N = amcl_->n(), maxn = 200, step = (N > maxn) ? N / maxn : 1;
    const auto& X = amcl_->xs(); const auto& Y = amcl_->ys(); const auto& T = amcl_->ths();
    for (int i = 0; i < N; i += step) {
      geometry_msgs::msg::Pose p;
      p.position.x = X[i]; p.position.y = Y[i];
      p.orientation.z = std::sin(T[i]/2); p.orientation.w = std::cos(T[i]/2);
      pa.poses.push_back(p);
    }
    parts_pub_->publish(pa);
  }

  void publishTf(const rclcpp::Time& t) {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = t; tf.header.frame_id = "map"; tf.child_frame_id = "odom";
    tf.transform.translation.x = tf_x_; tf.transform.translation.y = tf_y_;
    tf.transform.rotation.z = std::sin(tf_th_/2); tf.transform.rotation.w = std::cos(tf_th_/2);
    tf_br_->sendTransform(tf);
  }

  void publishMap(const rclcpp::Time& t) {
    nav_msgs::msg::OccupancyGrid g;
    g.header.stamp = t; g.header.frame_id = "map";
    g.info.resolution = res_; g.info.width = mapw_; g.info.height = maph_;
    g.info.origin.position.x = grid_->originX();
    g.info.origin.position.y = grid_->originY();
    g.info.origin.orientation.w = 1.0;
    std::vector<int8_t> data; grid_->toRosData(data);
    g.data = data;
    map_pub_->publish(g);
  }

  void publishPath(const rclcpp::Time& t) {
    nav_msgs::msg::Path path;
    path.header.stamp = t; path.header.frame_id = "map";
    for (const auto& nd : graph_.nodes()) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = path.header;
      ps.pose.position.x = nd.pose.x; ps.pose.position.y = nd.pose.y;
      ps.pose.orientation.z = std::sin(nd.pose.th/2); ps.pose.orientation.w = std::cos(nd.pose.th/2);
      path.poses.push_back(ps);
    }
    path_pub_->publish(path);
  }

  void publishGraph(const rclcpp::Time& t) {
    visualization_msgs::msg::MarkerArray ma;
    // nodes (spheres)
    visualization_msgs::msg::Marker nodes;
    nodes.header.stamp = t; nodes.header.frame_id = "map";
    nodes.ns = "nodes"; nodes.id = 0;
    nodes.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    nodes.action = visualization_msgs::msg::Marker::ADD;
    nodes.scale.x = nodes.scale.y = nodes.scale.z = 0.06;
    nodes.color.r = 0.1; nodes.color.g = 0.6; nodes.color.b = 1.0; nodes.color.a = 1.0;
    nodes.pose.orientation.w = 1.0;
    for (const auto& nd : graph_.nodes()) {
      geometry_msgs::msg::Point p; p.x = nd.pose.x; p.y = nd.pose.y; p.z = 0.02;
      nodes.points.push_back(p);
    }
    ma.markers.push_back(nodes);
    // odom edges (white) + loop edges (red)
    visualization_msgs::msg::Marker oe, le;
    oe.header = nodes.header; oe.ns = "odom_edges"; oe.id = 1;
    oe.type = visualization_msgs::msg::Marker::LINE_LIST;
    oe.scale.x = 0.012; oe.color.r=oe.color.g=oe.color.b=0.7; oe.color.a=0.6; oe.pose.orientation.w=1.0;
    le = oe; le.ns = "loop_edges"; le.id = 2; le.scale.x = 0.025;
    le.color.r = 1.0; le.color.g = 0.1; le.color.b = 0.1; le.color.a = 1.0;
    for (const auto& e : graph_.edges()) {
      geometry_msgs::msg::Point a, b;
      a.x = graph_.nodes()[e.a].pose.x; a.y = graph_.nodes()[e.a].pose.y; a.z = 0.02;
      b.x = graph_.nodes()[e.b].pose.x; b.y = graph_.nodes()[e.b].pose.y; b.z = 0.02;
      if (e.loop) { le.points.push_back(a); le.points.push_back(b); }
      else        { oe.points.push_back(a); oe.points.push_back(b); }
    }
    ma.markers.push_back(oe);
    ma.markers.push_back(le);
    graph_pub_->publish(ma);
  }

  // state
  std::mutex mtx_;
  std::unique_ptr<slam::OccupancyGrid> grid_;
  std::unique_ptr<slam::AMCL> amcl_;
  struct OdomS { double t, x, y, th; };
  std::deque<OdomS> odom_buf_;
  double ox_=0, oy_=0, oth_=0; bool odom_ready_=false;
  double po_x_=0, po_y_=0, po_th_=0; bool odom_init_=false;
  std::vector<float> prev_cloud_x_, prev_cloud_y_;
  double sx_=0, sy_=0, sth_=0;
  double tf_x_=0, tf_y_=0, tf_th_=0;
  double lmx_=0, lmy_=0, lmth_=0; bool first_write_=true; bool bootstrap_done_=false;
  bool localization_only_=false;   // true → preloaded saved map, never modified
  std::string map_path_;           // resolved saved-map yaml (localisation mode)
  double lx_=0, ly_=0, lyaw_=0; bool laser_ok_=false;
  int scan_count_=0;

  // params
  double res_; int mapw_, maph_;
  double range_min_, range_max_; int ray_step_;
  double quality_min_;
  double outlier_max_jump_; int map_pub_every_;
  std::string base_frame_;
  double scan_time_offset_;
  double min_dxy_, min_dth_, conf_thresh_, conf_rxy_, conf_rth_;
  bool sm_odom_en_; double sm_max_fit_, sm_max_jump_, sm_reject_;
  bool s2m_en_; double s2m_radius_; int s2m_minpts_;
  double s2m_reject_, s2m_maxcorr_, s2m_maxcorr_deg_, integ_max_fit_;
  double s2m_coarse_range_deg_, s2m_coarse_step_deg_;
  int bootstrap_cells_;
  // graph
  bool graph_en_; double kf_dist_, kf_angle_, loop_radius_, loop_min_fit_, graph_huber_;
  int loop_exclude_, graph_opt_every_;
  slam::PoseGraph graph_;
  double kf_x_=0, kf_y_=0, kf_th_=0; int last_kf_=-1;
  // backend thread
  std::thread backend_thread_;
  std::mutex be_mtx_;
  std::condition_variable be_cv_;
  bool backend_wanted_=false, backend_stop_=false;
  double init_x_, init_y_, init_th_, init_sxy_, init_sth_, tf_alpha_;
  bool publish_debug_=true; double scan_budget_ms_=80.0;

  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr parts_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr graph_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initpose_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_srv_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_br_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::TimerBase::SharedPtr tf_timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SlamNode>());
  rclcpp::shutdown();
  return 0;
}
