#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  // 5 system states: px, py, v, yaw, yaw_dot
  n_x_ = 5;
  // 2 states for noise 
  n_aug_ = n_x_ + 2;
  
  lambda_ = 3 - n_aug_;

  // weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  double k = lambda_ + n_aug_;
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    if (i == 0) {
      weights_(i) = lambda_ / k;
    } else {					
      weights_(i) = 1 / (2 * k);
    }
  }

  // predicted sigma points matrix will be a `n_x_ x (2 * n_aug_ + 1) ` matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // measurement covariance noise
  R_lidar_ = MatrixXd(2, 2);
  R_lidar_.fill(0.0);
  R_lidar_(0, 0) = std_laspx_ * std_laspx_;
  R_lidar_(1, 1) = std_laspy_ * std_laspy_;

  R_radar_ = MatrixXd(3, 3);
  R_radar_.fill(0.0);
  R_radar_(0, 0) = std_radr_ * std_radr_;
  R_radar_(1, 1) = std_radphi_ * std_radphi_;
  R_radar_(2, 2) = std_radrd_ * std_radrd_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(const MeasurementPackage& meas_package) {
  // for the first incoming measurement package: initialize the state vector
  if (!is_initialized_) {
    x_.fill(0.0);
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // initialize from laser measurement
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
      // v, yaw, yaw_dot cannot be inferred directly from px and py only
    } else {
      // initialize from radar measurement
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      x_(0) = cos(phi) * rho;
      x_(1) = sin(phi) * rho;
      // rho_dot is different from velocity: rho_dot is the radial velocity
      // and v is the tangential velocity
      // so no way to infer v, yaw, yaw_dot from rho, phi, rho_dot
    }
    // P_ initialized to be identity matrix
    P_.setIdentity();
    time_us_ = meas_package.timestamp_;
    
    is_initialized_ = true;

    std::cout<<"UKF initialized"<<std::endl;
    return;
  }

  // for the incoming packages, perform the predict-then-update process
  // check whether package shall be ignored
  if (!use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    return;
  }
  if (!use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    return;
  }

  double dt = (meas_package.timestamp_ - time_us_) / 1e6;
  // prediction
  std::cout<<"Doing prediction for type "<<meas_package.sensor_type_<<" after "<<dt<<" seconds"<<std::endl;
  Prediction(dt);

  // update
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    std::cout<<"Doing lidar update"<<std::endl;
    UpdateLidar(meas_package);
  } else {
    std::cout<<"Doing radar update"<<std::endl;
    UpdateRadar(meas_package);
  }

  time_us_ = meas_package.timestamp_;

  std::cout<<"x: "<<x_<<std::endl<<"P: "<<P_<<std::endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // augmented state vector x_aug
  VectorXd x_aug(n_aug_);
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;
  // augmented covariance matrix P_aug
  MatrixXd P_aug(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  MatrixXd L = P_aug.llt().matrixL();
  double k2 = sqrt(lambda_ + n_aug_);

  // compute sigma points
  MatrixXd sigma_points(n_aug_, 2 * n_aug_ + 1);
  sigma_points.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    VectorXd vec = k2 * L.col(i);
    sigma_points.col(i + 1)          = x_aug + vec;
    sigma_points.col(i + 1 + n_aug_) = x_aug - vec;
  }

  // do prediction using sigma points
  double dt2 = delta_t * delta_t;
  for (int i = 0; i < sigma_points.cols(); ++i) {
    VectorXd s = sigma_points.col(i);

    double px         = s(0);
    double py         = s(1);
    double v          = s(2);
    double yaw        = s(3);
    double yaw_dot    = s(4);
    
    double noise_a    = s(5);
    double noise_ydd  = s(6);

    // state delta defined by the process model;
    VectorXd dx(n_x_);
    dx.fill(0.0);

    // avoid zero division by checking raw_dot
    if (fabs(yaw_dot) < 1e-5) {
      // zero yaw_dot
      dx(0) = cos(yaw) * v * delta_t;
      dx(1) = sin(yaw) * v * delta_t;
    } else {
      // non-zero yaw_dot
      double k = v / yaw_dot;
      double yaw_kp1 = yaw + yaw_dot * delta_t;
      dx(0) = k * (sin(yaw_kp1) - sin(yaw));
      dx(1) = k * (-cos(yaw_kp1) + cos(yaw));
      dx(2) = 0;
      dx(3) = yaw_dot * delta_t;
      dx(4) = 0;
    }

    // state delta by the noise
    VectorXd noise_dx(n_x_);
    noise_dx.fill(0.0);

    noise_dx(0) = 0.5 * noise_a * dt2 * cos(yaw);
    noise_dx(1) = 0.5 * noise_a * dt2 * sin(yaw);
    noise_dx(2) = noise_a * delta_t;
    noise_dx(3) = 0.5 * noise_ydd * dt2;
    noise_dx(4) = noise_ydd * delta_t;
    
    Xsig_pred_.col(i) = s.head(n_x_) + dx + noise_dx;
  }
  
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(const MeasurementPackage& meas_package) {
  // calculate sigma points prediction, predicted mean and covariance matrix
  VectorXd x(n_x_);
  MatrixXd P(n_x_, n_x_);
  
  ComputePredictedMeanAndCovarianceMatrix(x, P);

  // use sigma points to generate sigma points in measurement space
  MatrixXd Zsig_pred(2, Xsig_pred_.cols());
  VectorXd z(2);
  MatrixXd S(2, 2);
  Zsig_pred.fill(0.0);
  z.fill(0.0);
  S.fill(0.0);

  // lidar measurement contains only px and py. Simply discarded the last 3 elements
  // of the vector and we can get the predicted measurement
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    Zsig_pred.col(i) = Xsig_pred_.col(i).head(2);
    z = z + weights_(i) * Zsig_pred.col(i);
  }
  for (int i = 0; i < Zsig_pred.cols(); ++i) {
    VectorXd diffz = Zsig_pred.col(i) - z;
    S = S + weights_(i) * diffz * diffz.transpose();
  }
  S = S + R_lidar_;
  
  // cross-correlation between sigma points in state space and measurement space
  MatrixXd T(n_x_, 2);
  T.fill(0.0);
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    T = T + weights_(i) * (Xsig_pred_.col(i) - x) * (Zsig_pred.col(i) - z).transpose();
  }

  // Kalman gain
  MatrixXd K = T * S.inverse();
  
  // update state
  VectorXd z_diff = meas_package.raw_measurements_ - z;
  x_ = x + K * z_diff;
  P_ = P - K * S * K.transpose();

  // NIS epsilon
  double epsilon = (z_diff.transpose() * S.inverse() * z_diff)(0,0);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const MeasurementPackage& meas_package) {
  // calculate sigma points prediction, predicted mean and covariance matrix
  // calculation is the same as when doing lidar update
  VectorXd x(n_x_);
  MatrixXd P(n_x_, n_x_);
  ComputePredictedMeanAndCovarianceMatrix(x, P);

  // use sigma points to generate sigma points in measurement space
  MatrixXd Zsig_pred(3, Xsig_pred_.cols());
  VectorXd z(3);
  MatrixXd S(3, 3);
  Zsig_pred.fill(0.0);
  z.fill(0.0);
  S.fill(0.0);

  // radar measurement contains rho, phi and rho_dot
  // rho can be inferred from px, py
  // phi can be inferred from px, py too
  // rho_dot is the projection of v onto rho
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    double px  = Xsig_pred_.col(i)(0);
    double py  = Xsig_pred_.col(i)(1);
    double v   = Xsig_pred_.col(i)(2);
    double yaw = Xsig_pred_.col(i)(3);

    double rho = sqrt(px * px + py * py);
    double phi = atan2(py, px);
    // normalizing phi
    while (phi >  M_PI) phi -= 2. * M_PI;
    while (phi < -M_PI) phi += 2. * M_PI;
    double rho_dot = (px * v * cos(yaw) + py * v * sin(yaw)) / rho;

    Zsig_pred.col(i) << rho, phi, rho_dot;
    z = z + weights_(i) * Zsig_pred.col(i);
  }
  for (int i = 0; i < Zsig_pred.cols(); ++i) {
    VectorXd diffz = Zsig_pred.col(i) - z;
    // normalizing phi
    while (diffz(1) >  M_PI) diffz(1) -= 2. * M_PI;
    while (diffz(1) < -M_PI) diffz(1) += 2. * M_PI;
    S = S + weights_(i) * diffz * diffz.transpose();
  }
  S = S + R_radar_;

  // cross-correlation between sigma points in state space and measurement space
  MatrixXd T(n_x_, 3);
  T.fill(0.0);
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    T = T + weights_(i) * (Xsig_pred_.col(i) - x) * (Zsig_pred.col(i) - z).transpose();
  }

  // Kalman gain
  MatrixXd K = T * S.inverse();

  // update state
  VectorXd z_diff = meas_package.raw_measurements_ - z;
  // normalizing angle
  while (z_diff(1) >  M_PI) z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;
  x_ = x + K * z_diff;
  P_ = P - K * S * K.transpose();

  // NIS epsilon
  double epsilon = (z_diff.transpose() * S.inverse() * z_diff)(0,0);
}

  /**
   * Computed predicted mean and covariance matrix
   */
void UKF::ComputePredictedMeanAndCovarianceMatrix(VectorXd& x, MatrixXd& P) {
  x.fill(0.0);
  P.fill(0.0);

  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    x = x + weights_(i) * Xsig_pred_.col(i);
  }
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    VectorXd diffx = Xsig_pred_.col(i) - x;
    // normalizing angle
    while (diffx(3) >  M_PI) diffx(3) -= 2. * M_PI;
    while (diffx(3) < -M_PI) diffx(3) += 2. * M_PI;
    P = P + weights_(i) * diffx * diffx.transpose();
  }
}
