//
// Created by Daniel Carlström Schad on 2022-05-20.
//

#pragma once

#include <vector>
#include <Eigen/Eigen>

using namespace Eigen;

class Fractal {
public:
    Fractal(Vector2d (*function)(Vector2d &, Vector2d &), Matrix2d (*jacobian)(Vector2d &, Matrix2d&), long double h = 1e-5,
            uint max_iter = 10000, long double comparison_eps = 1e-9, long double estimation_eps = 1e-15);

    explicit Fractal(Vector2d (*function)(Vector2d &, Vector2d &), long double h = 1e-5, uint max_iter = 10000,
                     long double comparison_eps = 1e-9, long double estimation_eps = 1e-15);

    // Implementation of newton's method in cases where analytic definition of the jacobian exists, and where
    // we have to estimate it using finite differences.
    Vector2d newtons_method(Vector2d initial_guess);

    // Get the index for the zero of a single initial guess to Newton's method
    int newton_index(Vector2d &x);

    // Implementation of Newton's method for linspaced grids of initial conditions
    MatrixXi newton_grid(std::vector<MatrixXd>& mesh);

    static std::vector<MatrixXd> create_mesh(int N, double a, double b, double c, double d);

private:
    std::function<Vector2d(Vector2d &, Vector2d &)> function;
    std::function<Matrix2d(Vector2d &, Matrix2d &)> jacobian;
    long double comparison_eps;
    long double estimation_eps;
    uint max_iter;
    long double h;
    std::vector<Vector2d> zeros = {};
    bool hasJacobian = false;

    Vector2d h1;
    Vector2d h2;
};