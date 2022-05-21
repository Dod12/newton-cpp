#include <iostream>
#include "Fractal.h"
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>
#include <pybind11/embed.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <cmath>

namespace py = pybind11;


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

Eigen::Tensor<double, 3, Eigen::RowMajor> getTensor(
        const pybind11::array_t<double>& inArray) {
    // request a buffer descriptor from Python
    pybind11::buffer_info buffer_info = inArray.request();

    // extract data an shape of input array
    auto *data = static_cast<double *>(buffer_info.ptr);
    std::vector<ssize_t> shape = buffer_info.shape;

    // wrap ndarray in Eigen::Map:
    // the second template argument is the rank of the tensor and has to be
    // known at compile time
    Eigen::TensorMap<Eigen::Tensor<double, 3, Eigen::RowMajor>> in_tensor(
            data, shape[0], shape[1], shape[2]);
    return in_tensor;
}

/*
Matrix<Vector<double, Dynamic>, Dynamic, Dynamic> getMatrix(
        const py::array& inArray) {

    Matrix<Vector<double, Dynamic>, Dynamic, Dynamic> out;
    out.setZero();

    std::vector<long> dims = inArray.request().shape;

    for (int i = 0; i < dims[1]; i++) {
        for (int j = 0; j < dims[2]; j++) {
            out(i, j) = VectorXd(inArray);
        }
    }


    return out;
}
*/

int main() {


    /*
    py::scoped_interpreter guard{};

    py::function setup_mesh = py::module_::import("src").attr("setup_mesh");

    py::tuple mesh = setup_mesh(100, -1, 1, -1, 1);

    //MatrixXd mesh1 = (py::array) mesh[0];

    std::cout << "Mesh: " << (std::string) repr(mesh) << std::endl;
    */
}