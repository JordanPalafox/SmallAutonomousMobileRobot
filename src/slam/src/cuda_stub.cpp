// cuda_stub.cpp — compiled INSTEAD of cuda_mcl.cu when CUDA isn't
// available.  Reports no GPU so AMCL uses its CPU scoring path.
namespace slam {

bool cuda_available() { return false; }

void cuda_score_particles(const float*, const float*, const float*, int,
                          const float*, const float*, int,
                          const float*, int, int, float,
                          float, float, float,
                          float, float, float,
                          float* /*out*/) {
  // never called when cuda_available() == false
}

}  // namespace slam
