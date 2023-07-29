add_library(
    hfLib
    # "${CMAKE_CURRENT_SOURCE_DIR}/src/hf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/input.hpp"
    )
target_include_directories(hf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(hf PUBLIC -DDISPERSION_VERSION="${DISPERSION_VERSION}")
