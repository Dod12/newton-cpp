cmake_minimum_required(VERSION 3.22)
project(Newton)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 2.9.2 REQUIRED)

INCLUDE_DIRECTORIES("/usr/local/include")

pybind11_add_module(py_newton src/py_newton.cpp src/Fractal.cpp src/Fractal.h)

target_link_libraries(py_newton PRIVATE Eigen3::Eigen)

install(TARGETS py_newton
        COMPONENT python
        LIBRARY DESTINATION "/Users/daniel/Documents/Projects/C/Newton/src"
        )

add_executable(Newton src/main.cpp src/Fractal.cpp src/Fractal.h src/py_newton.cpp)
target_link_libraries(Newton Eigen3::Eigen)
target_link_libraries(Newton pybind11::module pybind11::embed)
target_link_libraries(Newton Python::Python)

