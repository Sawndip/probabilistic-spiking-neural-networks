cmake_minimum_required(VERSION 3.13)

project(probabilistic_spiking_nn_modus_ponens VERSION 0.0.1)

include_directories(library/include
                    cereal/include)
add_subdirectory(library)

enable_testing()
add_subdirectory(test)