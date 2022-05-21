//
// Created by Daniel Carlström Schad on 2022-05-20.
//

#include "Fractal.h"
#include <Eigen/Eigen>
#include <iostream>

MatrixXd Fractal::estimate_jacobian(const VectorXd& x_0) {

    // We can estimate the jacobian by using finite differences
    long x_size = x_0.size();

    // Create jacobian of correct size
    MatrixXd j = MatrixXd::Zero(x_size, x_size);

    for (int i = 0; i < x_size; i++){
        VectorXd h = VectorXd::Zero(x_size);
        h(i) = 1e-5;
        VectorXd tmp = x_0 + h;
        j(all, i) = this->function(tmp);
    }

    return j;
}

VectorXd Fractal::newtons_method(VectorXd x) {

    MatrixXd j;
    int i = 0;
    int max_iter = 10000;

    while (1e-20 < this->function(x).norm() && i < max_iter) {
        if (this->hasJacobian) {
            j = this->jacobian(x);
        } else {
            j = estimate_jacobian(x);
        }
        if (j.determinant() == 0) {
            break;
        }
        x = x - j.inverse() * this->function(x);
    }
    return x;
}

Fractal::Fractal(VectorXd (*function)(VectorXd &)) {
    this->function = function;
    this->hasJacobian = false;
}

Fractal::Fractal(VectorXd (*function)(VectorXd &),
                 MatrixXd (*jacobian)(VectorXd &)) {
    this->function = function;
    this->jacobian = jacobian;
    this->hasJacobian = true;
}

Tensor<double, 3, RowMajor> Fractal::newton_grid(Tensor<double, 3, RowMajor> &initial_guesses) {

    auto dimensions = initial_guesses.dimensions();

    Tensor<double, 3, RowMajor> output(dimensions[0], dimensions[1], dimensions[2]);
    output.setZero();

    std::array<long, 3> len = {dimensions[0], 1, 1};
    std::array<long, 1> shape = {dimensions[0]};

    for (int i = 0; i < dimensions[1]; i++){
        for (int j = 0; j < dimensions[2]; j++){

        }
    }

    return output;
}
