#include "slam/amcl.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

// CUDA scoring entry point (defined in cuda_mcl.cu when built with CUDA;
// a weak CPU stub is provided below otherwise).
namespace slam {
bool cuda_available();
void cuda_score_particles(const float* xs, const float* ys, const float* ths, int n,
                          const float* cloud_x, const float* cloud_y, int m,
                          const float* lf, int W, int H, float res,
                          float ox, float oy, float sigma_hit,
                          float z_hit, float z_rand, float z_max_dist,
                          float* out_loglik);
}

namespace slam {

AMCL::AMCL(OccupancyGrid* grid, const AmclParams& p)
  : grid_(grid), p_(p), rng_(12345) {
  xs_.assign(p_.n_particles, 0.0f);
  ys_.assign(p_.n_particles, 0.0f);
  ths_.assign(p_.n_particles, 0.0f);
  ws_.assign(p_.n_particles, 1.0f / p_.n_particles);
  cuda_ok_ = p_.use_cuda && cuda_available();
}

void AMCL::initGaussian(double x, double y, double th, double sxy, double sth) {
  std::normal_distribution<double> nx(x, sxy), ny(y, sxy), nt(th, sth);
  for (int i = 0; i < p_.n_particles; ++i) {
    xs_[i] = static_cast<float>(nx(rng_));
    ys_[i] = static_cast<float>(ny(rng_));
    ths_[i] = static_cast<float>(wrap(nt(rng_)));
    ws_[i] = 1.0f / p_.n_particles;
  }
  cacheEstimate();
}

void AMCL::predict(double dx, double dy, double dth) {
  double trans = std::hypot(dx, dy);
  if (trans < 1e-9 && std::abs(dth) < 1e-9) return;
  double rot1 = (trans < 1e-9) ? 0.0 : std::atan2(dy, dx);
  double rot2 = dth - rot1;
  double sd_rot1  = std::sqrt(p_.alpha1 * rot1 * rot1 + p_.alpha2 * trans * trans);
  double sd_trans = std::sqrt(p_.alpha3 * trans * trans + p_.alpha4 * (rot1 * rot1 + rot2 * rot2));
  double sd_rot2  = std::sqrt(p_.alpha1 * rot2 * rot2 + p_.alpha2 * trans * trans);
  std::normal_distribution<double> g(0.0, 1.0);
  for (int i = 0; i < p_.n_particles; ++i) {
    double dr1 = rot1  - g(rng_) * sd_rot1;
    double dtr = trans - g(rng_) * sd_trans;
    double dr2 = rot2  - g(rng_) * sd_rot2;
    double head = ths_[i] + dr1;
    xs_[i] += static_cast<float>(dtr * std::cos(head));
    ys_[i] += static_cast<float>(dtr * std::sin(head));
    ths_[i] = static_cast<float>(wrap(ths_[i] + dr1 + dr2));
  }
}

void AMCL::update(const std::vector<float>& cx_in, const std::vector<float>& cy_in) {
  if (cx_in.empty()) return;
  // sub-sample
  std::vector<float> cx, cy;
  if (static_cast<int>(cx_in.size()) > p_.max_scan_pts) {
    int step = static_cast<int>(cx_in.size()) / p_.max_scan_pts;
    for (size_t i = 0; i < cx_in.size(); i += step) { cx.push_back(cx_in[i]); cy.push_back(cy_in[i]); }
  } else { cx = cx_in; cy = cy_in; }
  const int M = static_cast<int>(cx.size());

  const std::vector<float>& lf = grid_->likelihoodField(p_.sigma_hit);
  const int W = grid_->width(), H = grid_->height();
  const double res = grid_->resolution();
  const double ox = grid_->originX(), oy = grid_->originY();

  std::vector<float> loglik(p_.n_particles, 0.0f);

  if (cuda_ok_) {
    cuda_score_particles(xs_.data(), ys_.data(), ths_.data(), p_.n_particles,
                         cx.data(), cy.data(), M, lf.data(), W, H,
                         static_cast<float>(res), static_cast<float>(ox), static_cast<float>(oy),
                         static_cast<float>(p_.sigma_hit), static_cast<float>(p_.z_hit),
                         static_cast<float>(p_.z_rand), static_cast<float>(p_.z_max_dist),
                         loglik.data());
  } else {
    const double floor_v = std::exp(-(p_.z_max_dist * p_.z_max_dist) / (2.0 * p_.sigma_hit * p_.sigma_hit));
    const double rand_term = p_.z_rand / std::max(p_.z_max_dist, 1e-6);
    for (int i = 0; i < p_.n_particles; ++i) {
      double c = std::cos(ths_[i]), s = std::sin(ths_[i]);
      double ll = 0.0;
      for (int b = 0; b < M; ++b) {
        double wx = xs_[i] + c * cx[b] - s * cy[b];
        double wy = ys_[i] + s * cx[b] + c * cy[b];
        int gx = static_cast<int>(std::floor((wx - ox) / res));
        int gy = static_cast<int>(std::floor((wy - oy) / res));
        double v = (gx >= 0 && gx < W && gy >= 0 && gy < H)
                     ? lf[static_cast<size_t>(gy) * W + gx] : floor_v;
        ll += std::log(p_.z_hit * v + rand_term + 1e-12);
      }
      loglik[i] = static_cast<float>(ll);
    }
  }

  // normalise (subtract max, exp)
  float mx = *std::max_element(loglik.begin(), loglik.end());
  double total = 0.0;
  for (int i = 0; i < p_.n_particles; ++i) { ws_[i] = std::exp(loglik[i] - mx); total += ws_[i]; }
  if (total <= 0.0 || !std::isfinite(total)) {
    for (auto& w : ws_) w = 1.0f / p_.n_particles;
  } else {
    for (auto& w : ws_) w /= total;
  }
  cacheEstimate();
}

void AMCL::resample() {
  double neff_inv = 0.0;
  for (float w : ws_) neff_inv += static_cast<double>(w) * w;
  double neff = 1.0 / neff_inv;
  if (neff > p_.neff_ratio * p_.n_particles) { cacheEstimate(); return; }

  // low-variance resampler (no global random injection — mapping-safe)
  std::vector<float> nx(p_.n_particles), ny(p_.n_particles), nt(p_.n_particles);
  std::uniform_real_distribution<double> u(0.0, 1.0 / p_.n_particles);
  double r = u(rng_);
  std::vector<double> cum(p_.n_particles);
  cum[0] = ws_[0];
  for (int i = 1; i < p_.n_particles; ++i) cum[i] = cum[i-1] + ws_[i];
  cum[p_.n_particles - 1] = 1.0;
  int idx = 0;
  for (int i = 0; i < p_.n_particles; ++i) {
    double pos = r + static_cast<double>(i) / p_.n_particles;
    while (idx < p_.n_particles - 1 && cum[idx] < pos) ++idx;
    nx[i] = xs_[idx]; ny[i] = ys_[idx]; nt[i] = ths_[idx];
  }
  xs_ = nx; ys_ = ny; ths_ = nt;
  std::fill(ws_.begin(), ws_.end(), 1.0f / p_.n_particles);

  // Adaptive recovery: a deeply collapsed n_eff means the belief no longer
  // explains the scan — broaden it LOCALLY so the next sensor update can
  // re-acquire instead of staying stuck on a wrong, over-confident peak.
  if (p_.recovery_neff_ratio > 0.0 &&
      neff < p_.recovery_neff_ratio * p_.n_particles) {
    std::normal_distribution<double> jx(0.0, p_.recovery_sigma_xy);
    std::normal_distribution<double> jt(0.0, p_.recovery_sigma_theta);
    for (int i = 0; i < p_.n_particles; ++i) {
      xs_[i] += static_cast<float>(jx(rng_));
      ys_[i] += static_cast<float>(jx(rng_));
      ths_[i] = static_cast<float>(wrap(ths_[i] + jt(rng_)));
    }
  }
  cacheEstimate();
}

void AMCL::cacheEstimate() {
  double total = std::accumulate(ws_.begin(), ws_.end(), 0.0);
  if (total <= 0.0) return;
  double mx = 0, my = 0, ms = 0, mc = 0;
  for (int i = 0; i < p_.n_particles; ++i) {
    double w = ws_[i] / total;
    mx += w * xs_[i]; my += w * ys_[i];
    ms += w * std::sin(ths_[i]); mc += w * std::cos(ths_[i]);
  }
  est_ = { mx, my, std::atan2(ms, mc) };
}

double AMCL::certainty(double rxy, double rth) const {
  int in = 0;
  for (int i = 0; i < p_.n_particles; ++i) {
    double dxy = std::hypot(xs_[i] - est_.x, ys_[i] - est_.y);
    double dth = std::abs(wrap(ths_[i] - est_.th));
    if (dxy < rxy && dth < rth) ++in;
  }
  return static_cast<double>(in) / p_.n_particles;
}

}  // namespace slam
