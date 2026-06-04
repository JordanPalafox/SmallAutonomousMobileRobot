#include "slam/occupancy_grid.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstdlib>

namespace slam {

namespace {
// Trim ASCII whitespace from both ends.
std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}
// Directory part of a path (for resolving a relative PGM image path).
std::string dirOf(const std::string& p) {
  size_t s = p.find_last_of('/');
  return (s == std::string::npos) ? std::string() : p.substr(0, s);
}
// Read a binary P5 PGM (the format map_saver writes).  Skips '#' comment
// lines in the header.  pixels come back row 0 = TOP of image.
bool readPgmP5(const std::string& path, int& w, int& h,
               std::vector<unsigned char>& px, std::string& err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { err = "cannot open PGM " + path; return false; }
  std::string magic; f >> magic;
  if (magic != "P5") { err = "not a P5 PGM: " + path; return false; }
  auto nextInt = [&](long& out) -> bool {
    while (true) {
      int c = f.peek();
      if (c == EOF) return false;
      if (std::isspace(c)) { f.get(); continue; }
      if (c == '#') { std::string line; std::getline(f, line); continue; }
      break;
    }
    f >> out;
    return static_cast<bool>(f);
  };
  long ww, hh, maxv;
  if (!nextInt(ww) || !nextInt(hh) || !nextInt(maxv)) { err = "bad PGM header"; return false; }
  if (maxv > 255) { err = "16-bit PGM unsupported"; return false; }
  f.get();   // consume the single whitespace byte separating header from data
  w = static_cast<int>(ww); h = static_cast<int>(hh);
  px.resize(static_cast<size_t>(w) * h);
  f.read(reinterpret_cast<char*>(px.data()), static_cast<std::streamsize>(px.size()));
  if (static_cast<size_t>(f.gcount()) != px.size()) { err = "PGM data truncated"; return false; }
  return true;
}
}  // namespace

OccupancyGrid::OccupancyGrid(int width, int height, double resolution,
                             double l_occ, double l_free, double l_min, double l_max,
                             double display_l_occ, double display_l_free,
                             double occupied_stop)
  : w_(width), h_(height), res_(resolution),
    l_occ_(l_occ), l_free_(l_free), l_min_(l_min), l_max_(l_max),
    display_l_occ_(display_l_occ), display_l_free_(display_l_free),
    occupied_stop_(occupied_stop) {
  origin_x_ = -(w_ * res_) / 2.0;
  origin_y_ = -(h_ * res_) / 2.0;
  log_.assign(static_cast<size_t>(w_) * h_, 0.0f);
  lf_.assign(log_.size(), 0.0f);
}

void OccupancyGrid::reset() {
  std::fill(log_.begin(), log_.end(), 0.0f);
  lf_dirty_ = true;
}

bool OccupancyGrid::loadFromYaml(const std::string& yaml_path, std::string& err) {
  std::ifstream yf(yaml_path);
  if (!yf) { err = "cannot open yaml " + yaml_path; return false; }

  // ── Minimal nav2 map-yaml parser (flat keys; origin as block or inline
  //    list).  Robust enough for the files our map_saver writes. ──
  std::string image;
  double resolution = 0.0, ox = 0.0, oy = 0.0;
  int negate = 0;
  double occ_th = 0.65, free_th = 0.196;
  bool have_res = false, have_origin = false;
  std::vector<double> origin_vals;
  bool in_origin_block = false;
  std::string line;
  while (std::getline(yf, line)) {
    size_t hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    std::string t = trim(line);
    if (t.empty()) continue;
    if (in_origin_block) {
      if (t[0] == '-') { origin_vals.push_back(std::atof(trim(t.substr(1)).c_str())); continue; }
      in_origin_block = false;   // fall through and parse this line as a key
    }
    size_t colon = t.find(':');
    if (colon == std::string::npos) continue;
    std::string key = trim(t.substr(0, colon));
    std::string val = trim(t.substr(colon + 1));
    if (key == "image")                 image = val;
    else if (key == "resolution")     { resolution = std::atof(val.c_str()); have_res = true; }
    else if (key == "negate")           negate = std::atoi(val.c_str());
    else if (key == "occupied_thresh")  occ_th = std::atof(val.c_str());
    else if (key == "free_thresh")      free_th = std::atof(val.c_str());
    else if (key == "origin") {
      if (val.empty()) { in_origin_block = true; }     // block list on next lines
      else {                                            // inline [x, y, yaw]
        for (char& ch : val) if (ch == '[' || ch == ']' || ch == ',') ch = ' ';
        std::istringstream iss(val); double v;
        while (iss >> v) origin_vals.push_back(v);
      }
    }
  }
  if (origin_vals.size() >= 2) { ox = origin_vals[0]; oy = origin_vals[1]; have_origin = true; }
  if (image.empty())  { err = "yaml missing 'image'";       return false; }
  if (!have_res)      { err = "yaml missing 'resolution'";  return false; }
  if (!have_origin)   { err = "yaml missing 'origin'";      return false; }

  std::string img_path = image;
  if (img_path.empty() || img_path[0] != '/') img_path = dirOf(yaml_path) + "/" + img_path;

  int pw, ph;
  std::vector<unsigned char> px;
  if (!readPgmP5(img_path, pw, ph, px, err)) return false;

  // Geometry must match this grid: origin is derived from (w, h, res) at
  // construction, so a saved map built from a different grid config would
  // localise against a shifted world.  Refuse rather than silently offset.
  if (pw != w_ || ph != h_) {
    err = "size " + std::to_string(pw) + "x" + std::to_string(ph) +
          " != grid " + std::to_string(w_) + "x" + std::to_string(h_);
    return false;
  }
  if (std::abs(resolution - res_) > 1e-6) {
    err = "resolution " + std::to_string(resolution) + " != grid " + std::to_string(res_);
    return false;
  }
  if (std::abs(ox - origin_x_) > res_ || std::abs(oy - origin_y_) > res_) {
    err = "origin [" + std::to_string(ox) + "," + std::to_string(oy) +
          "] != grid [" + std::to_string(origin_x_) + "," + std::to_string(origin_y_) + "]";
    return false;
  }

  // Seed the log grid.  PGM row 0 = top of image; grid row 0 = bottom (ROS
  // convention) → flip vertically, exactly inverting the map_saver writer.
  for (int r = 0; r < ph; ++r) {
    const int gy = ph - 1 - r;
    for (int c = 0; c < pw; ++c) {
      const unsigned char p = px[static_cast<size_t>(r) * pw + c];
      const double prob = negate ? (p / 255.0) : (1.0 - p / 255.0);
      const size_t idx = static_cast<size_t>(gy) * w_ + c;
      if (prob >= occ_th)       log_[idx] = static_cast<float>(l_max_);   // wall
      else if (prob < free_th)  log_[idx] = static_cast<float>(l_min_);   // free
      else                      log_[idx] = 0.0f;                          // unknown
    }
  }
  lf_dirty_ = true;
  return true;
}

std::unique_ptr<OccupancyGrid> OccupancyGrid::cloneEmpty() const {
  return std::make_unique<OccupancyGrid>(w_, h_, res_, l_occ_, l_free_,
                                         l_min_, l_max_, display_l_occ_,
                                         display_l_free_, occupied_stop_);
}

void OccupancyGrid::adoptLogFrom(OccupancyGrid& other) {
  if (other.log_.size() != log_.size()) return;  // geometry mismatch guard
  log_ = std::move(other.log_);
  lf_dirty_ = true;
}

int OccupancyGrid::countOccupied() const {
  int n = 0;
  for (float v : log_) if (v > display_l_occ_) ++n;
  return n;
}

// Bresenham/DDA free-ray from (ox,oy) cell to (ex,ey) cell, voting free
// on traversed cells (excluding endpoint), stopping at strong walls.
void OccupancyGrid::integrateCloud(const Pose2& robot, const Cloud& world_ep,
                                   double laser_x, double laser_y) {
  if (world_ep.size() == 0) return;
  const double inv = 1.0 / res_;
  const double c = std::cos(robot.th), s = std::sin(robot.th);
  const double ox = robot.x + c * laser_x - s * laser_y;
  const double oy = robot.y + s * laser_x + c * laser_y;
  const int ox_c = static_cast<int>(std::floor((ox - origin_x_) * inv));
  const int oy_c = static_cast<int>(std::floor((oy - origin_y_) * inv));

  for (size_t i = 0; i < world_ep.size(); ++i) {
    const int ex_c = static_cast<int>(std::floor((world_ep.x[i] - origin_x_) * inv));
    const int ey_c = static_cast<int>(std::floor((world_ep.y[i] - origin_y_) * inv));

    // DDA from origin to endpoint, integer Bresenham.
    int x0 = ox_c, y0 = oy_c, x1 = ex_c, y1 = ey_c;
    int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;
    int cx = x0, cy = y0;
    while (true) {
      if (cx == x1 && cy == y1) break;  // stop before endpoint
      if (cx >= 0 && cx < w_ && cy >= 0 && cy < h_) {
        size_t idx = static_cast<size_t>(cy) * w_ + cx;
        // Stop the free-ray at established walls (don't erase them).
        if (log_[idx] > occupied_stop_) break;
        log_[idx] = std::max(l_min_, static_cast<double>(log_[idx]) + l_free_);
      }
      int e2 = 2 * err;
      if (e2 > -dy) { err -= dy; cx += sx; }
      if (e2 <  dx) { err += dx; cy += sy; }
    }
    // Occupied endpoint.
    if (ex_c >= 0 && ex_c < w_ && ey_c >= 0 && ey_c < h_) {
      size_t idx = static_cast<size_t>(ey_c) * w_ + ex_c;
      log_[idx] = std::min(l_max_, static_cast<double>(log_[idx]) + l_occ_);
    }
  }
  lf_dirty_ = true;
}

void OccupancyGrid::occupiedPoints(double rx, double ry, double radius,
                                   std::vector<float>& ox, std::vector<float>& oy,
                                   double threshold) const {
  if (threshold < 0.0) threshold = display_l_occ_;
  const double r2 = radius * radius;
  ox.clear(); oy.clear();
  // Scan only the cell window covering the query disc, not the whole grid.
  const double inv = 1.0 / res_;
  int x0 = std::max(0, static_cast<int>(std::floor((rx - radius - origin_x_) * inv)));
  int x1 = std::min(w_ - 1, static_cast<int>(std::floor((rx + radius - origin_x_) * inv)));
  int y0 = std::max(0, static_cast<int>(std::floor((ry - radius - origin_y_) * inv)));
  int y1 = std::min(h_ - 1, static_cast<int>(std::floor((ry + radius - origin_y_) * inv)));
  for (int yy = y0; yy <= y1; ++yy) {
    for (int xx = x0; xx <= x1; ++xx) {
      size_t idx = static_cast<size_t>(yy) * w_ + xx;
      if (log_[idx] > threshold) {
        double wx = origin_x_ + (xx + 0.5) * res_;
        double wy = origin_y_ + (yy + 0.5) * res_;
        double d2 = (wx - rx) * (wx - rx) + (wy - ry) * (wy - ry);
        if (d2 <= r2) { ox.push_back(static_cast<float>(wx)); oy.push_back(static_cast<float>(wy)); }
      }
    }
  }
}

// ── Felzenszwalb 1D squared-distance transform ──────────────────────
static void edt1d(const std::vector<float>& f, std::vector<float>& d, int n) {
  std::vector<int> v(n);
  std::vector<float> z(n + 1);
  int k = 0;
  v[0] = 0;
  z[0] = -std::numeric_limits<float>::infinity();
  z[1] =  std::numeric_limits<float>::infinity();
  for (int q = 1; q < n; ++q) {
    float s;
    while (true) {
      s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0f * (q - v[k]));
      if (s <= z[k]) --k; else break;
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = std::numeric_limits<float>::infinity();
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < q) ++k;
    float dq = q - v[k];
    d[q] = dq * dq + f[v[k]];
  }
}

void OccupancyGrid::buildLikelihoodField(double sigma) {
  const float INF = 1e20f;
  // Empty-map guard: with NO occupied cells the separable EDT computes
  // INF − INF = NaN, which would poison the AMCL sensor model.  Return
  // an all-zero field (every cell maximally far from any wall).
  int n_occ = 0;
  for (float v : log_) if (v > display_l_occ_) { ++n_occ; break; }
  if (n_occ == 0) {
    lf_.assign(log_.size(), 0.0f);
    lf_sigma_ = sigma; lf_dirty_ = false;
    return;
  }

  std::vector<float> dist(log_.size());
  // Seed: 0 at occupied cells, INF elsewhere (squared-distance domain in cells).
  for (size_t i = 0; i < log_.size(); ++i)
    dist[i] = (log_[i] > display_l_occ_) ? 0.0f : INF;

  // Columns then rows (separable EDT).
  std::vector<float> col(h_), cold(h_);
  for (int x = 0; x < w_; ++x) {
    for (int y = 0; y < h_; ++y) col[y] = dist[static_cast<size_t>(y) * w_ + x];
    edt1d(col, cold, h_);
    for (int y = 0; y < h_; ++y) dist[static_cast<size_t>(y) * w_ + x] = cold[y];
  }
  std::vector<float> row(w_), rowd(w_);
  for (int y = 0; y < h_; ++y) {
    for (int x = 0; x < w_; ++x) row[x] = dist[static_cast<size_t>(y) * w_ + x];
    edt1d(row, rowd, w_);
    for (int x = 0; x < w_; ++x) dist[static_cast<size_t>(y) * w_ + x] = rowd[x];
  }

  // Convert squared-cell-distance → gaussian likelihood in metres.
  const double res2 = res_ * res_;
  const double denom = 2.0 * sigma * sigma;
  lf_.resize(log_.size());
  for (size_t i = 0; i < log_.size(); ++i) {
    double d2_m = dist[i] * res2;                 // metres^2
    lf_[i] = static_cast<float>(std::exp(-d2_m / denom));
  }
  lf_sigma_ = sigma;
  lf_dirty_ = false;
}

const std::vector<float>& OccupancyGrid::likelihoodField(double sigma) {
  if (lf_dirty_ || std::abs(lf_sigma_ - sigma) > 1e-9)
    buildLikelihoodField(sigma);
  return lf_;
}

void OccupancyGrid::toRosData(std::vector<int8_t>& out) const {
  out.resize(log_.size());
  for (size_t i = 0; i < log_.size(); ++i) {
    // Use >= / <= (not strict >/<): log-odds accumulate in exact l_occ steps
    // (e.g. 0.50), so a freshly-confirmed wall lands EXACTLY on display_l_occ_
    // (2.5 = 5 hits).  With strict '>' that cell published as -1 (unknown/grey)
    // instead of 100 (occupied) — walls got fog/holes that poisoned the A* grid.
    if (log_[i] >= display_l_occ_)        out[i] = 100;
    else if (log_[i] <= display_l_free_)  out[i] = 0;
    else                                  out[i] = -1;
  }
}

}  // namespace slam
