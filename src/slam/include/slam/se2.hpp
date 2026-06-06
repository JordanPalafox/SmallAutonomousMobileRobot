// se2.hpp — SE(2) pose math + small structs shared across the SLAM node.
#pragma once
#include <cmath>
#include <vector>
#include <cstdint>
#include <utility>

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

// ── 2D rigid alignment (Umeyama/Kabsch, NO scale) for KNOWN correspondences ──
// Given N≥2 pairs (src_i → dst_i) find T such that dst ≈ R(θ)·src + t.
// The returned Pose2 T = {x=t_x, y=t_y, th=θ} is a DROP-IN for compose(T, p):
// compose(T, src_i) == R(θ)·src_i + t == dst_i, so it feeds anchorToAruco()
// directly. Closed-form (for 2D the atan2 form IS the SVD rotation, so no
// reflection edge case and no Eigen needed).
struct RigidFit { Pose2 T; double rmse{1e9}; int n{0}; bool ok{false}; };

inline RigidFit fit2dRigid(const std::vector<std::pair<double,double>>& src,
                           const std::vector<std::pair<double,double>>& dst) {
  RigidFit r; r.n = static_cast<int>(src.size());
  if (src.size() < 2 || src.size() != dst.size()) return r;
  const double inv = 1.0 / src.size();
  double sxm = 0, sym = 0, dxm = 0, dym = 0;
  for (size_t i = 0; i < src.size(); ++i) {
    sxm += src[i].first;  sym += src[i].second;
    dxm += dst[i].first;  dym += dst[i].second;
  }
  sxm *= inv; sym *= inv; dxm *= inv; dym *= inv;
  // 2×2 cross-covariance Σ = Σ (dst-d̄)(src-s̄)ᵀ
  double S00 = 0, S01 = 0, S10 = 0, S11 = 0;
  for (size_t i = 0; i < src.size(); ++i) {
    const double dsx = src[i].first - sxm, dsy = src[i].second - sym;
    const double dqx = dst[i].first - dxm, dqy = dst[i].second - dym;
    S00 += dqx * dsx; S01 += dqx * dsy;
    S10 += dqy * dsx; S11 += dqy * dsy;
  }
  const double th = std::atan2(S10 - S01, S00 + S11);
  const double c = std::cos(th), s = std::sin(th);
  r.T = { dxm - (c * sxm - s * sym), dym - (s * sxm + c * sym), wrap(th) };
  double sse = 0;
  for (size_t i = 0; i < src.size(); ++i) {
    const double px = c * src[i].first - s * src[i].second + r.T.x;
    const double py = s * src[i].first + c * src[i].second + r.T.y;
    const double ex = px - dst[i].first, ey = py - dst[i].second;
    sse += ex * ex + ey * ey;
  }
  r.rmse = std::sqrt(sse / src.size());
  r.ok = true;
  return r;
}

// Greedy drop-worst outlier rejection: fit all → drop the pair whose residual
// exceeds resid_thresh (refit) until none does or removing one would breach the
// min_inliers floor. Robust to a single mis-mapped marker by construction.
inline RigidFit fit2dRigidRobust(std::vector<std::pair<double,double>> src,
                                 std::vector<std::pair<double,double>> dst,
                                 double resid_thresh, int min_inliers) {
  RigidFit best;
  while (static_cast<int>(src.size()) >= min_inliers) {
    RigidFit f = fit2dRigid(src, dst);
    if (!f.ok) break;
    const double c = std::cos(f.T.th), s = std::sin(f.T.th);
    int worst = -1; double emax = -1.0;
    for (size_t i = 0; i < src.size(); ++i) {
      const double px = c * src[i].first - s * src[i].second + f.T.x;
      const double py = s * src[i].first + c * src[i].second + f.T.y;
      const double e = std::hypot(px - dst[i].first, py - dst[i].second);
      if (e > emax) { emax = e; worst = static_cast<int>(i); }
    }
    if (emax <= resid_thresh) { best = f; best.ok = true; break; }
    if (static_cast<int>(src.size()) - 1 < min_inliers) break;  // can't drop
    src.erase(src.begin() + worst);
    dst.erase(dst.begin() + worst);
  }
  return best;
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
