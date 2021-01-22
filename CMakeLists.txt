cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(StreamingSubmodularOptimization LANGUAGES CXX)
# find_package(OpenMP)

option(CREATE_EXPERIMENTS "Should experiments be created?" OFF)

# Include pybind11
# add_subdirectory(lib/pybind11)

###################################################################
# TARGET CONFIGURATION
###################################################################

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
# SET(CMAKE_CXX_EXTENSIONS OFF)
# SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native -mtune=native")

###################################################################
# TARGETS
###################################################################

include_directories("./include")

add_executable(run_kddcup ${SOURCES} experiments/kddcup99/run.cpp)

add_executable(main tests/main.cpp)

add_subdirectory(pybind11)
pybind11_add_module(PySSM include/Python.cpp)

if (CREATE_EXPERIMENTS)
    # target_link_libraries(threesieves-tests gtest gtest_main gcov)
    # target_compile_options(threesieves-tests PUBLIC -DTESTFILES_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/test-files/")
    # set_target_properties(threesieves-tests PROPERTIES LINKER_FLAGS "-fprofile-arcs -ftest-coverage")
endif ()
