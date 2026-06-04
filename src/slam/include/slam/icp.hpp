// icp.hpp — point-to-line scan matching with a spatial-hash NN index.
#pragma once
#include <vector>
#include <unordered_map>
#include "slam/se2.hpp"

namespace slam {

// Uniform spatial-hash nearest-neighbour index over a 2-D target cloud.
class NNIndex {
public:
  NNIndex(const std::vector<float>& x, const std::vector<float>& y, double cell);
  // nearest point to (qx,qy); returns index + squared distance. -1 if empty.
  int nearest(double qx, double qy, double& d2) const;
  // Append indices of all points in the 3×3 cell neighbourhood of (qx,qy).
  void neighborhood(double qx, double qy, std::vector<int>& out) const;
  size_t size() const { return x_.size(); }
  const std::vector<float>& x() const { return x_; }
  const std::vector<float>& y() const { return y_; }
private:
  long key(int cx, int cy) const { return static_cast<long>(cx) * 73856093L ^ static_cast<long>(cy) * 19349663L; }
  std::vector<float> x_, y_;
  double cell_;
  std::unordered_map<long, std::vector<int>> grid_;
};

struct IcpResult {
  double dx{0}, dy{0}, dth{0};
  double fitness{1e9};
  bool ok{false};
};

// Align src onto dst (both base/world frame), returning the SE(2) motion
// src→dst.  Point-to-line with progressively-narrowing reject distance.
IcpResult icpMatch(const std::vector<float>& src_x, const std::vector<float>& src_y,
                   const NNIndex& target, double init_dx, double init_dy, double init_dth,
                   int max_iter = 25, double reject = 0.4, int min_pts = 20);

}  // namespace slam
