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

    Network net7 = Network(200, 300, 100, 
                          fully_connected_init(), glorot_weights(generator));

    std::cout << "breakpoint" << std::endl;
}

int test(int argc, char** argv) {
    std::uint32_t n_input  = std::stoi(argv[1]);
    std::uint32_t n_hidden = std::stoi(argv[2]);
    std::uint32_t n_output = std::stoi(argv[3]);

    std::uint32_t ng_func_i = std::stoi(argv[4]);
    std::uint32_t w_func_i  = std::stoi(argv[5]);

    std::uint32_t expect_crash = std::stoi(argv[6]);

    double p = 0.5;
    double wa = 0.0;
    double wb = 1.0;

    if (argc > 7) {
        p = std::stod(argv[7]);
    } else if (argc > 8) {
        wa = std::stod(argv[8]);
        wb = std::stod(argv[9]);
    }

    int seed = 1337;
    std::default_random_engine generator;
    generator.seed(seed);

    NetworkGeneratorFunction rci;

    try {
        rci = random_connections_init(generator, p);
    } catch (std::invalid_argument&) {
        if (expect_crash)
            return 0;
        
        return -1;    
    }

    std::array<NetworkGeneratorFunction, 5> ng_funcs = {
        fully_connected_init(),
        perceptron_init_simple(),
        perceptron_init_hidden(),
        rci,
        // Special 0 connections network init function to test if it crashes
        // when 0 synapses are there in the end
        [](Neuron, Neuron) -> bool {return false;}
    };

    std::array<WeightInitializerFunction, 4> wei_funcs = {
        uniform_weights(generator, wa, wb), 
        normal_weights(generator, wa, wb), 
        glorot_weights(generator),
        // Special weight init function which always returns NaN
        [](NeuronList, Neuron, Neuron) -> double {return std::nan(""); }
    };

    try {
        Network net = Network(n_input, n_hidden, n_output, 
                              ng_funcs[ng_func_i], wei_funcs[w_func_i]);

        // std::cout << net;
    } catch (std::invalid_argument&) {
        if (expect_crash)
            return 0;
        
        return -1;
    } catch (std::logic_error&) {
        if (expect_crash)
            return 0;
        
        return -2;
    }

    return 0;
}

int main(int argc, char** argv) {
    if (argc == 1) {
        debug_test();
        return 0;
    }

    return test(argc, argv);
}