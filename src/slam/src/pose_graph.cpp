#include "slam/pose_graph.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>

namespace slam {

int PoseGraph::addNode(const Pose2& p, const std::vector<float>& sx, const std::vector<float>& sy) {
  GraphNode n; n.pose = p; n.scan_x = sx; n.scan_y = sy;
  if (nodes_.empty()) n.locked = true;     // anchor first node
  nodes_.push_back(std::move(n));
  return static_cast<int>(nodes_.size()) - 1;
}

void PoseGraph::addOdomEdge(int a, int b, const Pose2& meas) {
  double trans = std::hypot(meas.x, meas.y);
  double sxy = 0.02 * trans + 0.001;
  double sth = 0.03 * std::abs(meas.th) + 0.0087;
  edges_.push_back({a, b, meas, 1.0/(sxy*sxy), 1.0/(sth*sth), false});
}

void PoseGraph::addLoopEdge(int a, int b, const Pose2& meas, double sxy, double sth) {
  edges_.push_back({a, b, meas, 1.0/(sxy*sxy), 1.0/(sth*sth), true});
  ++n_loops_;
}

std::vector<int> PoseGraph::neighbours(int id, double radius, int exclude) const {
  std::vector<int> out;
  if (id < 0 || id >= size()) return out;
  double nx = nodes_[id].pose.x, ny = nodes_[id].pose.y;
  int cutoff = id - exclude;
  for (int i = 0; i < cutoff; ++i) {
    double d = std::hypot(nodes_[i].pose.x - nx, nodes_[i].pose.y - ny);
    if (d <= radius) out.push_back(i);
  }
  return out;
}

// residual r = log( meas⁻¹ · (a⁻¹·b) )
static Eigen::Vector3d edgeResidual(const Pose2& a, const Pose2& b, const Pose2& z) {
  Pose2 ab = compose(inverse(a), b);
  Pose2 e  = compose(inverse(z), ab);
  return Eigen::Vector3d(e.x, e.y, e.th);
}

bool PoseGraph::optimize(int max_iter, double huber, double& init_err, double& final_err) {
  const int N = size();
  if (N < 3 || edges_.empty()) return false;

  auto totalErr = [&](const std::vector<Pose2>& X) {
    double t = 0;
    for (const auto& e : edges_) {
      Eigen::Vector3d r = edgeResidual(X[e.a], X[e.b], e.meas);
      t += e.info_xy*(r(0)*r(0)+r(1)*r(1)) + e.info_th*r(2)*r(2);
    }
    return t;
  };

  std::vector<Pose2> X(N);
  for (int i = 0; i < N; ++i) X[i] = nodes_[i].pose;
  init_err = totalErr(X);
  double prev = init_err, lambda = 1e-3;

  for (int it = 0; it < max_iter; ++it) {
    std::vector<Eigen::Triplet<double>> trips;
    Eigen::VectorXd g = Eigen::VectorXd::Zero(3*N);

    for (const auto& e : edges_) {
      Eigen::Vector3d r = edgeResidual(X[e.a], X[e.b], e.meas);
      Eigen::Matrix3d O = Eigen::Matrix3d::Zero();
      O(0,0) = e.info_xy; O(1,1) = e.info_xy; O(2,2) = e.info_th;
      double e2 = r.transpose()*O*r;
      double w = (huber > 0 && e2 > huber*huber) ? huber/std::sqrt(e2) : 1.0;
      Eigen::Matrix3d Ow = O*w;
      // numerical Jacobians
      Eigen::Matrix3d Ja, Jb; const double eps = 1e-6;
      for (int k = 0; k < 3; ++k) {
        Pose2 ap = X[e.a];
        if (k==0) ap.x+=eps; else if (k==1) ap.y+=eps; else ap.th=wrap(ap.th+eps);
        Eigen::Vector3d ra = edgeResidual(ap, X[e.b], e.meas);
        Eigen::Vector3d da = ra - r; da(2)=wrap(da(2)); Ja.col(k)=da/eps;
        Pose2 bp = X[e.b];
        if (k==0) bp.x+=eps; else if (k==1) bp.y+=eps; else bp.th=wrap(bp.th+eps);
        Eigen::Vector3d rb = edgeResidual(X[e.a], bp, e.meas);
        Eigen::Vector3d db = rb - r; db(2)=wrap(db(2)); Jb.col(k)=db/eps;
      }
      Eigen::Matrix3d Haa=Ja.transpose()*Ow*Ja, Hbb=Jb.transpose()*Ow*Jb, Hab=Ja.transpose()*Ow*Jb;
      Eigen::Vector3d ba=Ja.transpose()*Ow*r, bb=Jb.transpose()*Ow*r;
      auto add=[&](int bi,int bj,const Eigen::Matrix3d& M){
        for(int i=0;i<3;++i)for(int j=0;j<3;++j) trips.emplace_back(3*bi+i,3*bj+j,M(i,j)); };
      add(e.a,e.a,Haa); add(e.b,e.b,Hbb); add(e.a,e.b,Hab); add(e.b,e.a,Hab.transpose());
      g.segment<3>(3*e.a)+=ba; g.segment<3>(3*e.b)+=bb;
    }
    for (int i=0;i<3*N;++i) trips.emplace_back(i,i,lambda);
    for (int i=0;i<N;++i) if (nodes_[i].locked)
      for(int k=0;k<3;++k){ trips.emplace_back(3*i+k,3*i+k,1e9); g(3*i+k)=0; }

    Eigen::SparseMatrix<double> H(3*N,3*N);
    H.setFromTriplets(trips.begin(), trips.end());
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver; solver.compute(H);
    if (solver.info()!=Eigen::Success) { lambda*=10; if(lambda>1e12)break; continue; }
    Eigen::VectorXd dx = solver.solve(-g);
    if (solver.info()!=Eigen::Success) { lambda*=10; if(lambda>1e12)break; continue; }

    std::vector<Pose2> Xt = X;
    for (int i=0;i<N;++i){ if(nodes_[i].locked) continue;
      Xt[i].x+=dx(3*i); Xt[i].y+=dx(3*i+1); Xt[i].th=wrap(Xt[i].th+dx(3*i+2)); }
    double ne = totalErr(Xt);
    if (ne < prev) {
      X = Xt; lambda = std::max(lambda*0.5,1e-9);
      double rel = (prev-ne)/std::max(prev,1e-12); prev = ne;
      if (rel < 1e-6) break;
    } else { lambda*=4; if(lambda>1e12)break; }
  }
  final_err = prev;
  if (final_err < init_err*0.999) {
    for (int i=0;i<N;++i) if(!nodes_[i].locked) nodes_[i].pose = X[i];
    return true;
  }
  return false;
}

}  // namespace slam
