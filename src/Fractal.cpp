//
// Created by Daniel Carlström Schad on 2022-05-20.
//

#include "Fractal.h"
#include "../extern/progressbar.hpp"
#include <Eigen/Eigen>
#include <iostream>


Vector2d Fractal::newtons_method(Vector2d x) {

    int i = 0;
    // Preallocate memory for outputs from functions
    Matrix2d j_n(x.size(), x.size());
    Vector2d f_n(x.size());

    if (this->hasJacobian) {
        this->function(x, f_n);
        while (this->estimation_eps < f_n.norm() && i < this->max_iter) {
            this->jacobian(x, j_n);
            if (j_n.determinant() == 0) {
                break;
            }
            x = x - j_n.inverse() * f_n;
            this->function(x, f_n);
        }
    } else {
        Vector2d x_tmp(x.size());
        this->function(x, f_n);
        while (this->estimation_eps < f_n.norm() && i < this->max_iter) {
            Vector2d tmp1 = x+this->h1;
            Vector2d tmp2 = x+this->h2;
            // Build determinant from finite differences
            this->function(tmp1, x_tmp);
            j_n(all, 0) = x_tmp;
            this->function(tmp2, x_tmp);
            j_n(all, 1) = x_tmp;

            if (j_n.determinant() == 0) {
                break;
            }
            x = x - j_n.inverse() * f_n;
            this->function(x, f_n);
        }
    }

    if (f_n.norm() > comparison_eps) {
        x << NAN, NAN;
    }

    return x;
}

Fractal::Fractal(Vector2d (*function)(Vector2d &, Vector2d &), long double h, uint max_iter,
                 long double comparison_eps, long double estimation_eps) {
    this->comparison_eps = comparison_eps;
    this->estimation_eps = estimation_eps;
    this->h1 = Vector2d {h, 0};
    this->h2 = Vector2d {0, h};
    this->max_iter = max_iter;
    this->function = function;
    this->hasJacobian = false;
}

Fractal::Fractal(Vector2d (*function)(Vector2d &, Vector2d &), Matrix2d (*jacobian)(Vector2d &, Matrix2d &), long double h,
                 uint max_iter, long double comparison_eps, long double estimation_eps) {
    this->comparison_eps = comparison_eps;
    this->estimation_eps = estimation_eps;
    this->h1 = Vector2d {h, 0};
    this->h2 = Vector2d {0, h};
    this->max_iter = max_iter;
    this->function = function;
    this->jacobian = jacobian;
    this->hasJacobian = true;
}

int Fractal::newton_index(Vector2d& x) {
    auto zero = this->newtons_method(x);
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
    MatrixXd y = VectorXd::LinSpaced(N, c, d).reverse().rowwise().replicate(N);

    return std::vector({x, y});
}

MatrixXi Fractal::newton_grid(std::vector<MatrixXd>& mesh) {

    Matrix<Vector2d, Dynamic, Dynamic> tmp(mesh[0].rows(), mesh[0].rows());

    initParallel();

    progressbar bar((int) tmp.cols());

    #pragma omp parallel for default(none) shared(mesh, tmp, bar)
    for (int i = 0; i < tmp.cols(); i++) {
        {
            for (int j = 0; j < tmp.rows(); j++) {
                Vector2d vec(2);
                vec << mesh[0](i, j), mesh[1](i, j);
                tmp(i, j) = this->newtons_method(vec);
            }
            #pragma omp critical
            bar.update();
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
