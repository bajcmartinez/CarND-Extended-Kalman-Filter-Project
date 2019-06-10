#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() = default;

KalmanFilter::~KalmanFilter() = default;

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z -  H_ * x_;
  UpdateStateWithY(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // load the variables from the vector
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  // calculate the angles
  float rho = sqrtf(px*px + py*py);
  float phi = atan2f(py, px);
  float rho_dot;

  // Normalize phi
  while (phi - z(1) > M_PI/2)
      phi -= M_PI;
  while (z(1) - phi  > M_PI/2)
      phi += M_PI;

  if (fabsf(rho) < 0.0001) {
    rho_dot = 0;
  } else {
    rho_dot = (px*vx + py*vy)/rho;
  }

  VectorXd h = VectorXd(3);
  h << rho, phi, rho_dot;
  VectorXd y = z - h;

  UpdateStateWithY(y);
}

void KalmanFilter::UpdateStateWithY(const VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = H_ * PHt + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;

  //new state
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
