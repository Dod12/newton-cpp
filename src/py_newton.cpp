//
// Created by Daniel Carlström Schad on 2022-05-21.
//

#define EIGEN_USE_MKL_ALL

#include <iostream>
#include "Fractal.h"
#include <Eigen/Eigen>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <cmath>

namespace py_funcs {
    Vector2d function1(Vector2d &x, Vector2d &out) {
        out(0) = pow(x(0), 3) - 3 * x(0) * pow(x(1), 2) - 1;
        out(1) = 3 * pow(x(0), 2) * x(1) - pow(x(1), 3);
        return out;
    }

    Matrix2d jacobian1(Vector2d &x, Matrix2d& out) {
        out(0,0) = 3 * pow(x(0), 2) - 3 * pow(x(1), 2);
        out(0,1) = -6 * x(0) * x(1);
        out(1, 0) = 6 * x(0) * x(1);
        out(1,1) = 3 * pow(x(0), 2) - 3 * pow(x(1), 2);
        return out;
    }

    Vector2d function3(Vector2d &x, Vector2d &out) {
        out(0) = pow(x(0),8) - 28 * pow(x(0),6) * pow(x(1),2) + 70 * pow(x(0),4) * pow(x(1), 4) + 15 * pow(x(0),4) - 28 * pow(x(0),2) * pow(x(1),6) - 90 * pow(x(0),2) * pow(x(1),2) + 15 * pow(x(1),4) - 16;
        out(1) = 8 * pow(x(0),7) * x(1) - 56 * pow(x(0),5) * pow(x(1),3) + 56 * pow(x(0),3) * pow(x(1),5) + 60 * pow(x(0),3) * x(1) - 8 * x(0) * pow(x(1),7) - 60 * x(0) * pow(x(1),3);
        return out;
    }

    void newton_grid(MatrixXd& x_mesh, MatrixXd& y_mesh, MatrixXi& indices, MatrixXi& iters, int function = 1) {

        auto fractal = Fractal(&py_funcs::function1, &py_funcs::jacobian1);

        if (function == 1) {
            std::cout << "Selecting function 1" << std::endl;
            fractal = Fractal(&py_funcs::function1, &py_funcs::jacobian1);
        } else if (function == 3) {
            std::cout << "Selecting function 3" << std::endl;
            fractal = Fractal(&py_funcs::function3);
        }

        auto mesh = std::vector({x_mesh, y_mesh});
        indices = fractal.newton_grid(mesh);
    }

    void get_funcs(std::function<Vector2d(Vector2d, Vector2d)> &f1) {
        std::cout << "Function found at " << &f1 << std::endl;
    }

    std::vector<MatrixXd> newton_mesh(int N, double a, double b, double c, double d) {
        return Fractal::create_mesh(N, a, b, c, d);
    }

    MatrixXi newton_meshgrid(int N, double a, double b, double c, double d, int function) {

        auto mesh = py_funcs::newton_mesh(N, a, b, c, d);
        MatrixXi indices(N,N);
        MatrixXi iters(N,N);
        py_funcs::newton_grid(mesh[0], mesh[1], indices, iters, function);
        return indices;
    }
}


PYBIND11_MODULE(py_newton, m) {
    m.doc() = "auto-compiled c++ extension";
    m.def("newton_grid", &py_funcs::newton_grid);
    m.def("get_func", &py_funcs::get_funcs);
    m.def("newton_mesh", &py_funcs::newton_mesh);
    m.def("newton_meshgrid", &py_funcs::newton_meshgrid);
}