// occupancy_grid.hpp — log-odds occupancy grid with DDA ray casting and
// a distance-transform likelihood field (for the AMCL sensor model).
#pragma once
#include <vector>
#include <cstdint>
#include <memory>
#include <string>
#include "slam/se2.hpp"

namespace slam {

class OccupancyGrid {
public:
  OccupancyGrid(int width, int height, double resolution,
                double l_occ = 0.5, double l_free = -0.2,
                double l_min = -2.0, double l_max = 4.0,
                double display_l_occ = 2.5, double display_l_free = -1.2,
                double occupied_stop = 3.0);

  // Integrate a deskewed scan: occupied endpoints + free-ray DDA.
  // robot pose is at scan time; ex/ey are world-frame endpoints.
  void integrateCloud(const Pose2& robot, const Cloud& world_endpoints,
                      double laser_x, double laser_y);

  // Occupied world-frame points within `radius` of (rx, ry).  Threshold
  // defaults to display_l_occ.
  void occupiedPoints(double rx, double ry, double radius,
                      std::vector<float>& out_x, std::vector<float>& out_y,
                      double threshold = -1.0) const;

  // Likelihood field: lf[idx] = exp(-d^2 / (2 sigma^2)), d = distance to
  // nearest occupied cell (metres).  Rebuilt lazily when dirty.
  const std::vector<float>& likelihoodField(double sigma);

  // Serialise to ROS OccupancyGrid.data (int8): 100/0/-1.
  void toRosData(std::vector<int8_t>& out) const;

  // Preload a saved nav2-format map (.yaml + .pgm) into the log grid, seeding
  // occupied cells to l_max and free cells to l_min (unknown → 0).  The exact
  // inverse of the map_saver writer.  Used for localisation against a known
  // map: AMCL/ICP match these cells and (in localisation-only mode) the grid
  // is never modified afterwards.  Returns false (and fills `err`) on any
  // failure — missing file, unreadable PGM, or geometry mismatch with this
  // grid (size/resolution/origin) — leaving the grid untouched.
  bool loadFromYaml(const std::string& yaml_path, std::string& err);

  void reset();

  // Fresh grid with identical geometry/log-odds config and an empty map —
  // used by the back-end to re-rasterise from keyframe scans off-thread.
  std::unique_ptr<OccupancyGrid> cloneEmpty() const;
  // Move `other`'s log-odds buffer into this grid (same geometry assumed)
  // and mark the likelihood field dirty.  Lets the back-end swap in a
  // re-mapped grid without invalidating the AMCL's grid pointer.
  void adoptLogFrom(OccupancyGrid& other);

  int width()  const { return w_; }
  int height() const { return h_; }
  double resolution() const { return res_; }
  double originX() const { return origin_x_; }
  double originY() const { return origin_y_; }
  double displayLOcc() const { return display_l_occ_; }
  int countOccupied() const;

  // Raw log-odds access (read).
  const std::vector<float>& log() const { return log_; }

private:
  void buildLikelihoodField(double sigma);

  int w_, h_;
  double res_, origin_x_, origin_y_;
  double l_occ_, l_free_, l_min_, l_max_;
  double display_l_occ_, display_l_free_, occupied_stop_;

  std::vector<float> log_;     // h_*w_

  // Likelihood field cache
  std::vector<float> lf_;
  double lf_sigma_{-1.0};
  bool lf_dirty_{true};
};

}  // namespace slam
