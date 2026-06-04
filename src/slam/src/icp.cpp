#include "slam/icp.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <limits>

namespace slam {

NNIndex::NNIndex(const std::vector<float>& x, const std::vector<float>& y, double cell)
  : x_(x), y_(y), cell_(cell) {
  for (size_t i = 0; i < x_.size(); ++i) {
    int cx = static_cast<int>(std::floor(x_[i] / cell_));
    int cy = static_cast<int>(std::floor(y_[i] / cell_));
    grid_[key(cx, cy)].push_back(static_cast<int>(i));
  }
}

int NNIndex::nearest(double qx, double qy, double& best_d2) const {
  best_d2 = std::numeric_limits<double>::infinity();
  int best = -1;
  if (x_.empty()) return -1;
  int cx = static_cast<int>(std::floor(qx / cell_));
  int cy = static_cast<int>(std::floor(qy / cell_));
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      auto it = grid_.find(key(cx + dx, cy + dy));
      if (it == grid_.end()) continue;
      for (int idx : it->second) {
        double ddx = x_[idx] - qx, ddy = y_[idx] - qy;
        double d2 = ddx * ddx + ddy * ddy;
        if (d2 < best_d2) { best_d2 = d2; best = idx; }
      }
    }
  }
  return best;
}

void NNIndex::neighborhood(double qx, double qy, std::vector<int>& out) const {
  out.clear();
  int cx = static_cast<int>(std::floor(qx / cell_));
  int cy = static_cast<int>(std::floor(qy / cell_));
  for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy) {
      auto it = grid_.find(key(cx + dx, cy + dy));
      if (it == grid_.end()) continue;
      out.insert(out.end(), it->second.begin(), it->second.end());
    }
}

// Per-target-point normal via PCA on its k nearest neighbours, gathered from
// the spatial-hash 3×3 cell window (O(n·k) overall, not O(n²)).
static void estimateNormals(const NNIndex& idx,
                            std::vector<float>& nx, std::vector<float>& ny) {
  const auto& X = idx.x();
  const auto& Y = idx.y();
  const size_t n = X.size();
  nx.assign(n, 0.0f); ny.assign(n, 0.0f);
  const int K = 6;
  const double R2 = 0.09;             // 30 cm neighbourhood
  std::vector<int> cand;
  std::vector<std::pair<double,int>> near;
  for (size_t i = 0; i < n; ++i) {
    idx.neighborhood(X[i], Y[i], cand);
    near.clear();
    for (int j : cand) {
      double dx = X[j] - X[i], dy = Y[j] - Y[i];
      double dd = dx * dx + dy * dy;
      if (dd < R2) near.emplace_back(dd, j);
    }
    if (near.size() < 3) { nx[i] = 0; ny[i] = 1; continue; }
    int kk = std::min<int>(K, static_cast<int>(near.size()));
    std::partial_sort(near.begin(), near.begin() + kk, near.end());
    double mx = 0, my = 0;
    for (int a = 0; a < kk; ++a) { mx += X[near[a].second]; my += Y[near[a].second]; }
    mx /= kk; my /= kk;
    double sxx = 0, sxy = 0, syy = 0;
    for (int a = 0; a < kk; ++a) {
      double dx = X[near[a].second] - mx, dy = Y[near[a].second] - my;
      sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
    }
    sxx /= kk; sxy /= kk; syy /= kk;
    // smallest-eigenvector of [[sxx,sxy],[sxy,syy]] = surface normal
    double tr = sxx + syy, det = sxx * syy - sxy * sxy;
    double lam = tr / 2.0 - std::sqrt(std::max(0.0, tr * tr / 4.0 - det));
    double vx = sxy, vy = lam - sxx;
    double nn = std::sqrt(vx * vx + vy * vy);
    if (nn < 1e-9) { nx[i] = 0; ny[i] = 1; } else { nx[i] = vx / nn; ny[i] = vy / nn; }
  }
}

IcpResult icpMatch(const std::vector<float>& sx, const std::vector<float>& sy,
                   const NNIndex& target, double dx, double dy, double dth,
                   int max_iter, double reject, int min_pts) {
  IcpResult res;
  if (sx.size() < static_cast<size_t>(min_pts) || target.size() < static_cast<size_t>(min_pts))
    { res.dx = dx; res.dy = dy; res.dth = dth; return res; }

  std::vector<float> nx, ny;
  estimateNormals(target, nx, ny);
  const auto& TX = target.x();
  const auto& TY = target.y();

  double x = dx, y = dy, th = dth;
  for (int it = 0; it < max_iter; ++it) {
    double c = std::cos(th), s = std::sin(th);
    double frac = static_cast<double>(it) / std::max(max_iter - 1, 1);
    double cur_reject = reject * (1.0 - 0.6 * frac);
    double rj2 = cur_reject * cur_reject;

    // Build normal equations for point-to-line: minimise Σ (n·(R p + t − q))²
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    int inl = 0;
    double sse = 0.0;
    for (size_t i = 0; i < sx.size(); ++i) {
      double px = c * sx[i] - s * sy[i] + x;
      double py = s * sx[i] + c * sy[i] + y;
      double d2;
      int j = target.nearest(px, py, d2);
      if (j < 0 || d2 > rj2) continue;
      double Nx = nx[j], Ny = ny[j];
      double qx = TX[j], qy = TY[j];
      // residual along normal
      double r = Nx * (qx - px) + Ny * (qy - py);
      // Jacobian wrt (tx, ty, dth): d(px)/dth = -py_rel... use current pt
      double Jx = Nx;
      double Jy = Ny;
      double Jth = Nx * (-py) + Ny * (px);  // ∂(n·p)/∂θ with p rotated
      A(0,0) += Jx*Jx; A(0,1) += Jx*Jy; A(0,2) += Jx*Jth;
      A(1,1) += Jy*Jy; A(1,2) += Jy*Jth;
      A(2,2) += Jth*Jth;
      b(0) += Jx * r; b(1) += Jy * r; b(2) += Jth * r;
      sse += d2; ++inl;
    }
    if (inl < min_pts) break;
    A(1,0) = A(0,1); A(2,0) = A(0,2); A(2,1) = A(1,2);
    Eigen::Vector3d delta = A.ldlt().solve(b);
    double tx = delta(0), ty = delta(1), dthi = delta(2);
    // apply increment (use temporaries so y doesn't read the new x)
    double cc = std::cos(dthi), ss = std::sin(dthi);
    double nxp = cc * x - ss * y + tx;
    double nyp = ss * x + cc * y + ty;
    x = nxp; y = nyp;
    th = wrap(th + dthi);
    res.fitness = std::sqrt(sse / inl);
    if (std::abs(tx) < 1e-5 && std::abs(ty) < 1e-5 && std::abs(dthi) < 1e-5) break;
  }
  res.dx = x; res.dy = y; res.dth = th; res.ok = true;
  return res;
}

}  // namespace slam
