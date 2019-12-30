cmake_minimum_required(VERSION 3.13)

project(probabilistic_spiking_nn_modus_ponens VERSION 0.0.1)

# Treat warnings as errors. Use newest standard.
# This is called two times. Once for the library and
# once for tests.
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -Werror -std=c++17")

include_directories(library)
add_subdirectory(library)

enable_testing()
add_subdirectory(test)

# Treat warnings as errors. Use newest standard.
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -Werror -std=c++17")