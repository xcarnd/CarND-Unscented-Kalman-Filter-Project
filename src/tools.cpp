#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // size of the two input must be the same
  assert(estimations.size() == ground_truth.size());
  // contains at least one element
  assert(estimations.size() > 0);

  VectorXd ret(estimations[0].size());
  ret.fill(0.0);
  for (int i = 0; i < estimations.size(); ++i) {
    VectorXd diff = estimations[i] - ground_truth[i];
    ret.array() += (diff.array() * diff.array());
  }
  ret /= estimations.size();
  return ret.array().sqrt();
}
