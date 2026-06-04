// pose_graph.hpp — SE(2) pose graph + Levenberg-Marquardt optimiser
// with a Huber robust kernel, for loop-closure SLAM.
#pragma once
#include <vector>
#include "slam/se2.hpp"

namespace slam {

struct GraphNode {
  Pose2 pose;
  std::vector<float> scan_x, scan_y;   // base-frame keyframe scan
  bool locked = false;
};

struct GraphEdge {
  int a, b;
  Pose2 meas;          // relative pose b-in-a frame
  double info_xy, info_th;   // diagonal information (1/σ²)
  bool loop = false;
};

class PoseGraph {
public:
  int addNode(const Pose2& p, const std::vector<float>& sx, const std::vector<float>& sy);
  void addOdomEdge(int a, int b, const Pose2& meas);
  void addLoopEdge(int a, int b, const Pose2& meas, double sxy, double sth);

  int size() const { return static_cast<int>(nodes_.size()); }
  int loopCount() const { return n_loops_; }
  const std::vector<GraphNode>& nodes() const { return nodes_; }
  const std::vector<GraphEdge>& edges() const { return edges_; }
  GraphNode& node(int i) { return nodes_[i]; }

  // Candidate older nodes within `radius` of node `id`, excluding the
  // most-recent `exclude` nodes.
  std::vector<int> neighbours(int id, double radius, int exclude) const;

  // Run LM optimisation; returns true if error was reduced.
  bool optimize(int max_iter, double huber, double& init_err, double& final_err);

  void reset() { nodes_.clear(); edges_.clear(); n_loops_ = 0; }

private:
  std::vector<GraphNode> nodes_;
  std::vector<GraphEdge> edges_;
  int n_loops_ = 0;
};

}  // namespace slam
