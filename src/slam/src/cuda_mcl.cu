// cuda_mcl.cu — GPU particle scoring for the AMCL sensor model.
//
// One CUDA thread per particle: transforms every beam of the sub-sampled
// scan into the particle's world frame, looks up the likelihood field,
// and accumulates Σ log(z_hit·lf + z_rand_floor).  The likelihood field
// and the scan are copied to the device once per update() call.
//
// Deployment target is the Jetson Nano (Maxwell, sm_53, 128 cores): even
// that weak GPU scores a few hundred particles × 240 beams comfortably
// inside the 5.5 Hz frame budget, keeping the quad-A57 CPU free for the
// scan-matching front-end and the graph back-end.  On a dev RTX laptop
// (sm_86) it is effectively free, so the particle count can be raised for
// a tighter belief.

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

namespace slam {

__global__ void score_kernel(const float* xs, const float* ys, const float* ths, int n,
                             const float* cx, const float* cy, int m,
                             const float* lf, int W, int H, float inv_res,
                             float ox, float oy, float floor_v,
                             float z_hit, float rand_term, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float c = cosf(ths[i]), s = sinf(ths[i]);
  float px = xs[i], py = ys[i];
  float ll = 0.0f;
  for (int b = 0; b < m; ++b) {
    float wx = px + c * cx[b] - s * cy[b];
    float wy = py + s * cx[b] + c * cy[b];
    int gx = (int)floorf((wx - ox) * inv_res);
    int gy = (int)floorf((wy - oy) * inv_res);
    float v = (gx >= 0 && gx < W && gy >= 0 && gy < H) ? lf[(size_t)gy * W + gx] : floor_v;
    ll += logf(z_hit * v + rand_term + 1e-12f);
  }
  out[i] = ll;
}

// Persistent device buffers (resized as needed) to avoid per-call churn.
static float *d_xs=nullptr,*d_ys=nullptr,*d_ths=nullptr,*d_out=nullptr;
static float *d_cx=nullptr,*d_cy=nullptr,*d_lf=nullptr;
static int cap_n=0, cap_m=0, cap_lf=0;
static bool g_checked=false, g_ok=false;

bool cuda_available() {
  if (g_checked) return g_ok;
  g_checked = true;
  int cnt = 0;
  cudaError_t e = cudaGetDeviceCount(&cnt);
  g_ok = (e == cudaSuccess && cnt > 0);
  if (g_ok) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
      fprintf(stderr, "[slam] CUDA MCL active on: %s (sm_%d%d)\n",
              prop.name, prop.major, prop.minor);
  }
  return g_ok;
}

void cuda_score_particles(const float* xs, const float* ys, const float* ths, int n,
                          const float* cloud_x, const float* cloud_y, int m,
                          const float* lf, int W, int H, float res,
                          float ox, float oy, float sigma_hit,
                          float z_hit, float z_rand, float z_max_dist,
                          float* out_loglik) {
  if (n > cap_n) {
    cudaFree(d_xs); cudaFree(d_ys); cudaFree(d_ths); cudaFree(d_out);
    cudaMalloc(&d_xs, n*sizeof(float)); cudaMalloc(&d_ys, n*sizeof(float));
    cudaMalloc(&d_ths, n*sizeof(float)); cudaMalloc(&d_out, n*sizeof(float));
    cap_n = n;
  }
  if (m > cap_m) {
    cudaFree(d_cx); cudaFree(d_cy);
    cudaMalloc(&d_cx, m*sizeof(float)); cudaMalloc(&d_cy, m*sizeof(float));
    cap_m = m;
  }
  int lfsize = W*H;
  if (lfsize > cap_lf) {
    cudaFree(d_lf); cudaMalloc(&d_lf, lfsize*sizeof(float)); cap_lf = lfsize;
  }

  cudaMemcpy(d_xs, xs, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ys, ys, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ths, ths, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cx, cloud_x, m*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cy, cloud_y, m*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lf, lf, lfsize*sizeof(float), cudaMemcpyHostToDevice);

  float floor_v = expf(-(z_max_dist*z_max_dist) / (2.0f*sigma_hit*sigma_hit));
  float rand_term = z_rand / fmaxf(z_max_dist, 1e-6f);
  int threads = 128, blocks = (n + threads - 1) / threads;
  score_kernel<<<blocks, threads>>>(d_xs, d_ys, d_ths, n, d_cx, d_cy, m,
                                     d_lf, W, H, 1.0f/res, ox, oy, floor_v,
                                     z_hit, rand_term, d_out);
  cudaMemcpy(out_loglik, d_out, n*sizeof(float), cudaMemcpyDeviceToHost);
}

}  // namespace slam
