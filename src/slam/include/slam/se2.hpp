// se2.hpp — SE(2) pose math + small structs shared across the SLAM node.
#pragma once
#include <cmath>
#include <vector>
#include <cstdint>

namespace slam {

inline double wrap(double a) {
  while (a >  M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

struct Pose2 {
  double x{0.0}, y{0.0}, th{0.0};
};

// a ⊕ b  (compose: apply b in a's frame)
inline Pose2 compose(const Pose2& a, const Pose2& b) {
  const double c = std::cos(a.th), s = std::sin(a.th);
  return { a.x + c * b.x - s * b.y,
           a.y + s * b.x + c * b.y,
           wrap(a.th + b.th) };
}

inline Pose2 inverse(const Pose2& a) {
  const double c = std::cos(a.th), s = std::sin(a.th);
  return { -c * a.x - s * a.y,
            s * a.x - c * a.y,
           wrap(-a.th) };
}

// relative pose of `to` expressed in `from` frame
inline Pose2 relative(const Pose2& from, const Pose2& to) {
  return compose(inverse(from), to);
}

// A 2-D scan as flat (x,y) point pairs in some frame.
struct Cloud {
  std::vector<float> x;
  std::vector<float> y;
  size_t size() const { return x.size(); }
  void clear() { x.clear(); y.clear(); }
  void reserve(size_t n) { x.reserve(n); y.reserve(n); }
  void push(float px, float py) { x.push_back(px); y.push_back(py); }
};

}  // namespace slam
