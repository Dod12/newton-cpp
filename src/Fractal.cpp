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
        VectorXd h_vec = VectorXd::Zero(x_size);
        h_vec(i) = (double) this->h;
        VectorXd tmp = x_0 + h_vec;
        j(all, i) = this->function(tmp);
    }

    return j;
}

VectorXd Fractal::newtons_method(VectorXd x) {

    MatrixXd j;
    int i = 0;

    while (this->estimation_eps < this->function(x).norm() && i < this->max_iter) {
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

Fractal::Fractal(VectorXd (*function)(VectorXd &), long double h, uint max_iter,
                 long double comparison_eps, long double estimation_eps) {
    this->comparison_eps = comparison_eps;
    this->estimation_eps = estimation_eps;
    this->h = h;
    this->max_iter = max_iter;
    this->function = function;
    this->hasJacobian = false;
}

Fractal::Fractal(VectorXd (*function)(VectorXd &), MatrixXd (*jacobian)(VectorXd &), long double h,
                 uint max_iter, long double comparison_eps, long double estimation_eps) {
    this->comparison_eps = comparison_eps;
    this->estimation_eps = estimation_eps;
    this->h = h;
    this->max_iter = max_iter;
    this->function = function;
    this->jacobian = jacobian;
    this->hasJacobian = true;
}

int Fractal::newton_index(VectorXd& x) {
    auto zero = this->newtons_method(x);
    if (this->function(zero).norm() > comparison_eps) {
        zero << NAN, NAN;
    }
    for (int k = 0; k < this->zeros.size(); k++) {
        if (zero.isApprox(this->zeros[k], (double) this->comparison_eps) || (zero.hasNaN() && this->zeros[k].hasNaN())) {
            return k;
        }
    }
    this->zeros.push_back(zero);
    return (int) this->zeros.size() - 1;
}

std::vector<MatrixXd> Fractal::create_mesh(int N, double a, double b, double c, double d) {

    MatrixXd x = VectorXd::LinSpaced(N, a, b).transpose().colwise().replicate(N);
    MatrixXd y = VectorXd::LinSpaced(N, c, b).reverse().rowwise().replicate(N);

    return std::vector({x, y});
}

MatrixXi Fractal::newton_grid(std::vector<MatrixXd>& mesh) {

    Matrix<VectorXd, Dynamic, Dynamic> tmp(mesh[0].rows(), mesh[0].rows());

    initParallel();

    #pragma omp parallel for default(none) shared(mesh, tmp)
    for (int i = 0; i < tmp.cols(); i++) {
        {
            for (int j = 0; j < tmp.cols(); j++) {
                VectorXd vec(2);
                vec << mesh[0](i, j), mesh[1](i, j);
                tmp(i, j) = this->newtons_method(vec);
            }
        }
    }


    // Cannot multithread this because of the race condition for the zeros attribute.

    MatrixXi out(tmp.rows(), tmp.cols());
    out.setZero();

    for (int i = 0; i < tmp.cols(); i++) {
        for (int j = 0; j < tmp.cols(); j++) {
            out(i, j) = this->newton_index(tmp(i, j));
        }
    }

    return out;
}
