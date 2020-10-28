cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(StreamingSubmodularOptimization LANGUAGES CXX)
find_package(OpenMP)

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
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

###################################################################
# TARGETS
###################################################################

include_directories("./include")

add_executable(run_kddcup ${SOURCES} experiments/kddcup99/run.cpp)
# if (CREATE_EXPERIMENTS)
#     # target_link_libraries(threesieves-tests gtest gtest_main gcov)
#     # target_compile_options(threesieves-tests PUBLIC -DTESTFILES_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/test-files/")
#     # set_target_properties(threesieves-tests PROPERTIES LINKER_FLAGS "-fprofile-arcs -ftest-coverage")
# endif ()

add_executable(main tests/main.cpp)

add_subdirectory(pybind11)
pybind11_add_module(PySSM include/Python.cpp)
# target_link_libraries(PySSM PRIVATE ${MY_LIBRARIES})

# SET(PYTHON_COMMON_MODULE_NAME threesieves)
# SET(PYBIND11_CPP_STANDARD -std=c++17)
# SET(SOURCES
#         src/functions/ivm/InformativeVectorMachine.cpp
#         src/optimizers/random/RandomOptimizer.cpp
#         src/optimizers/threesieves/ThreeSieves.cpp
#         src/optimizers/greedy/GreedyOptimizer.cpp
#         src/optimizers/salsa/SalsaSinglePass.cpp
#         src/optimizers/sievestreaming/Sieve.cpp
#         src/optimizers/sievestreaming/SieveStreaming.cpp
#         src/optimizers/sievestreaming-pp/SieveStreamingPP.cpp)

# pybind11_add_module(${PYTHON_COMMON_MODULE_NAME} ${SOURCES} src/Python.cpp)
# if (OpenMP_CXX_FOUND)
#     target_link_libraries(${PYTHON_COMMON_MODULE_NAME} PRIVATE OpenMP::OpenMP_CXX)
# endif ()

###################################################################
# TARGETS TEST
###################################################################

# if (CREATE_TESTS)
#     set(gtest_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/lib/googletest/googletest")
#     add_subdirectory(${gtest_src_dir})
#     include_directories(${gtest_src_dir}/include ${gtest_src_dir})

#     add_executable(threesieves-tests ${SOURCES} tests/Tests.cpp tests/CSVFile.cpp)

#     target_link_libraries(threesieves-tests gtest gtest_main gcov)
#     target_compile_options(threesieves-tests PUBLIC -DTESTFILES_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/test-files/")
#     set_target_properties(threesieves-tests PROPERTIES LINKER_FLAGS "-fprofile-arcs -ftest-coverage")
# endif ()