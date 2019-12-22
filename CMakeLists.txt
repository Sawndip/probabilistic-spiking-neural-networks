cmake_minimum_required(VERSION 3.13)

project(probabilistic_spiking_nn_modus_ponens VERSION 0.0.1)

include_directories(library)
add_subdirectory(library)

enable_testing()
add_subdirectory(test)

# Treat warnings as errors.
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -Wall -Werror)