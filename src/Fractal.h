//
// Created by Daniel Carlström Schad on 2022-05-20.
//

#pragma once

#include <vector>
#include <Eigen/Eigen>

using namespace Eigen;

class Fractal {
public:
    Fractal(VectorXd (*function)(VectorXd &), MatrixXd (*jacobian)(VectorXd &), long double h = 1e-5,
            uint max_iter = 10000, long double comparison_eps = 1e-9, long double estimation_eps = 1e-15);

    explicit Fractal(VectorXd (*function)(VectorXd &), long double h = 1e-5, uint max_iter = 10000,
                     long double comparison_eps = 1e-9, long double estimation_eps = 1e-15);

    // Implementation of newton's method in cases where analytic definition of the jacobian exists, and where
    // we have to estimate it using finite differences.
    VectorXd newtons_method(VectorXd initial_guess);

    // Get the index for the zero of a single initial guess to Newton's method
    int newton_index(VectorXd &x);

    // Implementation of Newton's method for linspaced grids of initial conditions
    MatrixXi newton_grid(std::vector<MatrixXd>& mesh);

    static std::vector<MatrixXd> create_mesh(int N, double a, double b, double c, double d);

private:
    std::function<VectorXd(VectorXd &)> function;
    std::function<MatrixXd(VectorXd &)> jacobian;
    long double comparison_eps;
    long double estimation_eps;
    uint max_iter;
    long double h;
    std::vector<VectorXd> zeros = {};
    bool hasJacobian = false;

    // Helper functions for computations
    MatrixXd estimate_jacobian(const VectorXd& x);
};