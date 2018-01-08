#include <math.h>
#include <iostream>
#include "Eigen-3.3/Eigen/Core"

// Helper functions
constexpr double pi() { return M_PI; };
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

const double Lf = 2;

/*
  Return the next state
  Note: state is [x, y, psi, vel]
  Note: actuators are [delta, accel]
*/
Eigen::VectorXd globalKinematic(Eigen::VectorXd state,
                                Eigen::VectorXd actuators,
                                double dt){
  // Create a new vector for the next state
  Eigen::VectorXd next_state(state.size());

  auto x     = state(0);
  auto y     = state(1);
  auto psi   = state(2);
  auto vel   = state(3);
  auto delta = actuators(0);
  auto accel = actuators(1);

  /*
    Model equations:
    x_[t+1] = x[t] + vel[t] * cos(psi[t]) * dt;
    y_[t+1] = y[t] + vel[t] * sin(psi[t]) * dt;
    psi_[t+1] = psi + vel / Lf * delta * dt;
    v_[t+1] = v[t] + accel * dt;
  */
  next_state(0) = x + vel * cos(psi) * dt;
  next_state(1) = y + vel * sin(psi) * dt;
  next_state(2) = psi + vel / Lf * delta * dt;
  next_state(3) = vel + accel * dt;

  return next_state;
}

int main(){
  // [x, y, psi, vel]
  Eigen::VectorXd state(4);
  // [detla, accel]
  Eigen::VectorXd actuators(2);

  state << 0, 0, deg2rad(45), 1;
  actuators << deg2rad(5), 1;

  auto next_state = globalKinematic(state, actuators, 0.3);

  std::cout << next_state << std::endl;
}
