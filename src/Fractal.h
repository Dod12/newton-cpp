//
// Created by Daniel Carlström Schad on 2022-05-20.
//

#ifndef NEWTON_FRACTAL_H
#define NEWTON_FRACTAL_H

#include <vector>
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

class Fractal {
public:
    Fractal(VectorXd (*function)(VectorXd &),
            MatrixXd (*jacobian)(VectorXd &));

    explicit Fractal(VectorXd (*function)(VectorXd &));

    // Implementation of newton's method in cases where analytic definition of the jacobian exists, and where
    // we have to estimate it using finite differences.
    VectorXd newtons_method(VectorXd initial_guess);

    Tensor<double, 3, RowMajor> newton_grid(Tensor<double, 3, RowMajor>& initial_guesses);

private:
    MatrixXd estimate_jacobian(const VectorXd& x);
    std::function<VectorXd(VectorXd &)> function;
    std::function<MatrixXd(VectorXd &)> jacobian;
    bool hasJacobian = false;
};

#endif //NEWTON_FRACTAL_H
