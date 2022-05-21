//
// Created by Daniel Carlström Schad on 2022-05-21.
//

#include <iostream>
#include "Fractal.h"
#include <Eigen/Eigen>
#include <pybind11/embed.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <cmath>


VectorXd function(VectorXd &x) {
    double f1 = pow(x(0),3) - 3*x(0)*pow(x(1),2) - 1;
    double f2 = 3*pow(x(0),2)*x(1) - pow(x(1),3);
    return Vector2d(f1, f2);
}

MatrixXd jacobian(VectorXd &x) {
    double f1x1 = 3*pow(x(0),2) - 3*pow(x(1),2);
    double f1x2 = -6*x(0)*x(1);
    double f2x1 = 6*x(0)*x(1);
    double f2x2 = 3*pow(x(0),2) - 3*pow(x(1),2);
    return Matrix2d {{f1x1, f1x2}, {f2x1, f2x2}};
}


Vector2d guess_newton(Ref<Vector2d> x) {
    Fractal fractal = Fractal(&function, &jacobian);
    VectorXd initial_guess = (VectorXd) Vector2d{10., 10.};
    auto zero = fractal.newtons_method((VectorXd) initial_guess);
    return zero;
}


PYBIND11_MODULE(py_newton, m) {
m.doc() = "auto-compiled c++ extension";
m.def("newton", &guess_newton);
}