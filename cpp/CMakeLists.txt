cmake_minimum_required(VERSION 3.13)
project(salientime)
set(CMAKE_CXX_STANDARD 14)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(pybind11)

pybind11_add_module(select_k select_k.cpp)
target_compile_features(select_k PUBLIC cxx_std_14)
set_target_properties(select_k PROPERTIES SUFFIX ".so")