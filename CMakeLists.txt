cmake_minimum_required(VERSION 3.22)
project(Newton)

set(CMAKE_CXX_STANDARD 20)
set(PYTHONVERSION 3.9 CACHE STRING "")
set(PYBIND11_PYTHON_VERSION 3.9 CACHE STRING "")
set(PYBIND11_FINDPYTHON OFF)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(MKL_LINK dynamic)
    set(MKL_INTERFACE lp64)
    find_package(MKL CONFIG REQUIRED)
    add_compile_options(-m64  -I"${MKL_ROOT}/include -fopenmp")
    add_link_options(${MKL_ROOT}/lib/libmkl_intel_lp64.a ${MKL_ROOT}/lib/libmkl_intel_thread.a ${MKL_ROOT}/lib/libmkl_core.a -lpthread -lm -ldl) # -liomp5 Can't get OpenMP to work
    add_definitions(-DEIGEN_USE_MKL_ALL)
endif()

find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Python3 3.9 COMPONENTS Interpreter Development)
find_package(pybind11 2.9.2 REQUIRED CONFIG)

include_directories("/usr/local/include")

pybind11_add_module(py_newton src/py_newton.cpp src/Fractal.cpp src/Fractal.h)

target_link_libraries(py_newton PRIVATE Eigen3::Eigen)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(py_newton PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(py_newton PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(py_newton PUBLIC $<LINK_ONLY:MKL::MKL>)
    target_link_libraries(py_newton PUBLIC OpenMP)
endif()

add_executable(Newton src/main.cpp src/Fractal.cpp src/Fractal.h src/py_newton.cpp)
target_link_libraries(Newton PUBLIC Eigen3::Eigen pybind11::module pybind11::embed)
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(Newton PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_include_directories(Newton PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    target_link_libraries(Newton PUBLIC $<LINK_ONLY:MKL::MKL>)
    target_link_libraries(Newton PUBLIC OpenMP::OpenMP_CXX)
endif()