#include "core/include/network.h"

#include <iostream>
#include <random>

void debug_test() {
    std::default_random_engine generator;
    generator.seed(1337);

    Network net1 = Network(2, 0, 1,
                           perceptron_init_simple, 
                           glorot_weights(generator));
    std::cout << "breakpoint";

    Network net2 = Network(2, 0, 1,
                        perceptron_init_simple, 
                        normal_weights(generator));
    std::cout << "breakpoint";

    Network net3 = Network(2, 0, 1,
                    perceptron_init_simple, 
                    uniform_weights(generator));
    std::cout << "breakpoint";

    Network net4 = Network(2, 3, 1, perceptron_init_hidden, glorot_weights(generator));

    std::cout << "breakpoint";

    Network net5 = Network(2, 3, 1, fully_connected_init, glorot_weights(generator));

    std::cout << "breakpoint";
}

int main(int argc, char** argv) {
    debug_test();
}