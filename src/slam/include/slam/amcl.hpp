// amcl.hpp — Adaptive Monte Carlo Localisation (particle filter).
#pragma once
#include <vector>
#include <random>
#include "slam/se2.hpp"
#include "slam/occupancy_grid.hpp"

namespace slam {

struct AmclParams {
  int n_particles = 600;
  double alpha1 = 0.10, alpha2 = 0.08, alpha3 = 0.10, alpha4 = 0.05;
  double sigma_hit = 0.10, z_hit = 0.85, z_rand = 0.15, z_max_dist = 2.0;
  int max_scan_pts = 240;
  double neff_ratio = 0.5;
  // Recovery jitter: when n_eff collapses below recovery_neff_ratio·N the
  // belief has effectively lost the robot, so broaden the cloud locally
  // (Gaussian about each surviving particle) to re-acquire.  This is local
  // — NOT global scatter — so it stays safe during from-scratch mapping.
  double recovery_neff_ratio = 0.15;
  double recovery_sigma_xy = 0.10;
  double recovery_sigma_theta = 0.08;
  bool use_cuda = true;
};

class AMCL {
public:
  AMCL(OccupancyGrid* grid, const AmclParams& p);

  void initGaussian(double x, double y, double th, double sxy, double sth);

  // odom_delta is the body-frame motion since last step.
  void predict(double dxb, double dyb, double dth);
  // sensor update against grid likelihood field; cloud in base frame.
  void update(const std::vector<float>& cloud_x, const std::vector<float>& cloud_y);
  void resample();

  Pose2 estimate() const { return est_; }
  double certainty(double rxy, double rth) const;

  int n() const { return p_.n_particles; }
  const std::vector<float>& xs() const { return xs_; }
  const std::vector<float>& ys() const { return ys_; }
  const std::vector<float>& ths() const { return ths_; }

  bool cudaActive() const { return cuda_ok_; }

private:
  void cacheEstimate();

  OccupancyGrid* grid_;
  AmclParams p_;
  std::vector<float> xs_, ys_, ths_, ws_;
  Pose2 est_;
  std::mt19937 rng_;
  bool cuda_ok_ = false;
};

}  // namespace slam
