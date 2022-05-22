//
// Created by Daniel Carlström Schad on 2022-05-21.
//

//EIGEN_USE_MKL_ALL = true

#include <iostream>
#include "Fractal.h"
#include <Eigen/Eigen>
#include <pybind11/embed.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <cmath>

namespace py_funcs {
    VectorXd function(VectorXd &x) {
        double f1 = pow(x(0), 3) - 3 * x(0) * pow(x(1), 2) - 1;
        double f2 = 3 * pow(x(0), 2) * x(1) - pow(x(1), 3);
        return Vector2d(f1, f2);
    }

    MatrixXd jacobian(VectorXd &x) {
        double f1x1 = 3 * pow(x(0), 2) - 3 * pow(x(1), 2);
        double f1x2 = -6 * x(0) * x(1);
        double f2x1 = 6 * x(0) * x(1);
        double f2x2 = 3 * pow(x(0), 2) - 3 * pow(x(1), 2);
        return Matrix2d{{f1x1, f1x2},
                        {f2x1, f2x2}};
    }

    MatrixXi newton_grid(MatrixXd& x_mesh, MatrixXd& y_mesh) {
        Fractal fractal = Fractal(&py_funcs::function, &py_funcs::jacobian);
        auto mesh = std::vector({x_mesh, y_mesh});
        return fractal.newton_grid(mesh);
    }

    std::vector<MatrixXd> newton_mesh(int N, double a, double b, double c, double d) {
        return Fractal::create_mesh(N, a, b, c, d);
    }

    MatrixXi newton_meshgrid(int N, double a, double b, double c, double d) {
        auto mesh = py_funcs::newton_mesh(N, a, b, c, d);
        return py_funcs::newton_grid(mesh[0], mesh[1]);
    }
}


PYBIND11_MODULE(py_newton, m) {
    m.doc() = "auto-compiled c++ extension";
    m.def("newton_grid", &py_funcs::newton_grid);
    m.def("newton_mesh", &py_funcs::newton_mesh);
    m.def("newton_meshgrid", &py_funcs::newton_meshgrid);
}