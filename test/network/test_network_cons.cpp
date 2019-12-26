#include "core/include/network.h"

#include <iostream>
#include <random>

// Called in debugger. Used only during active development.
void debug_test() {
    std::default_random_engine generator;
    generator.seed(1337);

    Network net1 = Network(2, 0, 1,
                           perceptron_init_simple(), 
                           glorot_weights(generator));
    std::cout << "breakpoint";

    Network net2 = Network(2, 0, 1,
                        perceptron_init_simple(), 
                        normal_weights(generator));
    std::cout << "breakpoint";

    Network net3 = Network(2, 0, 1,
                    perceptron_init_simple(), 
                    uniform_weights(generator));
    std::cout << "breakpoint";

    Network net4 = Network(2, 3, 1, perceptron_init_hidden(), glorot_weights(generator));

    std::cout << "breakpoint";

    Network net5 = Network(2, 3, 1, fully_connected_init(), glorot_weights(generator));

    std::cout << "breakpoint" << std::endl;

    Network net6 = Network(2, 3, 1, random_connections_init(generator, 0.5), 
                                    glorot_weights(generator));

    std::cout << "breakpoint" << std::endl;

    std::cout << net6;
}

int main(int argc, char** argv) {
    if (argc == 1) {
        debug_test();
    }
}