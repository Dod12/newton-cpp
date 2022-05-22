#include <iostream>
#include "Fractal.h"
#include <Eigen/Eigen>
#include <cstdlib>


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

void print_mesh(Matrix<VectorXd, Dynamic, Dynamic>& result) {
    MatrixXd res1(result.rows(), result.cols()), res2(result.rows(), result.cols());
    res1.setZero();
    res2.setZero();

    for (int i = 0; i < res1.rows(); i++) {
        for (int j = 0; j < res1.cols(); j++) {
            double x_i = result(i, j)(0);
            double y_j = result(i, j)(1);
            res1(i, j) = x_i;
            res2(i, j) = y_j;
        }
    }

    std::cout << "X1:\n" << res1 << std::endl;
    std::cout << "X2:\n" << res2 << std::endl;
}

int main(int argc, char *argv[]) {

    // Set defaults
    int N = 100;
    double a = -1;
    double b =  1;
    double c = -1;
    double d =  1;

    for (int i = 0; i < argc; i++) {
        if (strcmp(*(argv + i), "-n") == 0) {
            std::cout << "N: " << *(argv + i + 1) << std::endl;
            N = (int) strtol(*(argv + i + 1), nullptr, 0);
        } else if (strcmp(*(argv + i), "-a") == 0) {
            std::cout << "a: " << *(argv + i + 1) << std::endl;
            a = (int) strtod(*(argv + i + 1), nullptr);
        } else if (strcmp(*(argv + i), "-b") == 0) {
            std::cout << "b: " << *(argv + i + 1) << std::endl;
            b = (int) strtod(*(argv + i + 1), nullptr);
        } else if (strcmp(*(argv + i), "-c") == 0) {
            std::cout << "c: " << *(argv + i + 1) << std::endl;
            c = (int) strtod(*(argv + i + 1), nullptr);
        } else if (strcmp(*(argv + i), "-d") == 0) {
            std::cout << "d: " << *(argv + i + 1) << std::endl;
            d = (int) strtod(*(argv + i + 1), nullptr);
        } else {

        }
    }


    auto start = std::chrono::steady_clock::now();

    auto result = Fractal::create_mesh(N, a, b, c, d);

    auto stop = std::chrono::steady_clock::now();

    auto diff = stop - start;

    std::cout << "Initialization time: " << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;

    start = std::chrono::steady_clock::now();

    Fractal fractal = Fractal(&function, &jacobian);
    auto res = fractal.newton_grid(result);

    stop = std::chrono::steady_clock::now();

    diff = stop - start;

    std::cout << "Execution time: " << std::chrono::duration <double> (diff).count() << " s" << std::endl;

    return 0;
}

MatrixXi newton_grid(int N, double a, double b, double c, double d) {
    Fractal fractal = Fractal(&function, &jacobian);
    auto mesh = Fractal::create_mesh(N, a, b, c, d);
    return fractal.newton_grid(mesh);
}