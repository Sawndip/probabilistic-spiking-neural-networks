#include <core/network.h>

#include <random>
#include <iostream>

void debug_run() {
    std::default_random_engine generator;

    Network net(2, 3, 1, fully_connected_init(), glorot_weights(generator));

    SignalList test_signals;
    Signal s1 = Signal::from_string("^^__^^__");
    Signal s2 = Signal::from_string("__^^__^^");
    test_signals.add(s1);
    test_signals.add(s2);
    test_signals.equalize_lengths();

    SignalList output = net.forward(test_signals, generator);

    std::cout << test_signals << std::endl;

    std::cout << output << std::endl;
}

int test_run(int argc, char** argv) {
    std::uint32_t n_input  = std::stoi(argv[1]);
    std::uint32_t n_hidden = std::stoi(argv[2]);
    std::uint32_t n_output = std::stoi(argv[3]);

    std::uint32_t ng_func_i = std::stoi(argv[4]);
    std::uint32_t w_func_i  = std::stoi(argv[5]);

    std::uint32_t expect_crash = std::stoi(argv[6]);

    double p = 0.5;
    double wa = 0.0;
    double wb = 1.0;

    double tau1 = 5;
    double tau2 = 13;
    double tau_m = 3;

    if (argc > 7) {
        p = std::stod(argv[7]);
    } else if (argc > 8) {
        wa = std::stod(argv[8]);
        wb = std::stod(argv[9]);
    } else if (argc > 10) {
        tau1 = std::stod(argv[10]);
        tau2 = std::stod(argv[11]);
        tau_m = std::stod(argv[12]);
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
        rci
    };

    std::array<WeightInitializerFunction, 4> wei_funcs = {
        uniform_weights(generator, wa, wb), 
        normal_weights(generator, wa, wb), 
        glorot_weights(generator)
    };

    try {
        Network net = Network(n_input, n_hidden, n_output, 
                              ng_funcs[ng_func_i], 
                              wei_funcs[w_func_i],
                              default_exponential_kernels(4, tau1, tau2, tau_m));

        SignalList test_signals;
        Signal s1 = Signal::from_string("^^__^^__");
        Signal s2 = Signal::from_string("__^^__^^");
        test_signals.add(s1);
        test_signals.add(s2);
        test_signals.equalize_lengths();

        SignalList output = net.forward(test_signals, generator);

        std::cout << test_signals << std::endl;

        // Print output only if small so test times won't suffer due to printing
        if (output.cdata().size() < 5)
            std::cout << output << std::endl;    
    } catch (std::invalid_argument& ex) {
        std::cout << ex.what() << std::endl;

        if (expect_crash)
            return 0;
        
        return -1;
    } catch (std::logic_error& ex) {
        std::cout << ex.what() << std::endl;

        if (expect_crash)
            return 0;
        
        return -2;
    }

    return 0;
}

int main(int argc, char** argv) {
    if (argc == 1) {
        debug_run();
        return 0;
    }

    return test_run(argc, argv);
}