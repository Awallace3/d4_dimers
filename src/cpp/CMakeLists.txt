cmake_minimum_required(VERSION 3.10)

project(DISPERSION VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fopenmp -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -fopenmp")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(APPLE)
    set(CMAKE_C_COMPILER "gcc")
    set(CMAKE_CXX_COMPILER "g++-12")
    include_directories(/usr/local/include
        /usr/local/include/eigen3
        /usr/include
        /usr/lib
        ./extern/pybind11/include
        )
    link_directories(/usr/local/lib
        /usr/local/include/eigen3
        /usr/include
        /usr/lib
        )
endif()
if(LINUX)
    set(CMAKE_CXX_COMPILER "gcc")
    set(CMAKE_C_COMPILER= "g++-11")
    include_directories(/usr/local/include
        /usr/include/eigen3
        /usr/include
        /usr/lib
        ./extern/pybind11/include
        )
    link_directories(/usr/local/lib
        /usr/include/eigen3
        /usr/include
        /usr/lib
        )
endif()


find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(MPI REQUIRED)

add_subdirectory(src)
