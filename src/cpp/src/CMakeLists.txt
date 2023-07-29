add_executable(disp disp.cpp input.cpp helper.cpp example.cpp)
target_include_directories(disp PUBLIC ${CMAKE_SOURCE_DIR}/include
    /usr/local/include
    /usr/local/include/eigen3
     /usr/include/eigen3
     /usr/include
     ../extern/pybind11/include
    )
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
