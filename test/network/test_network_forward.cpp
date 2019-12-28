#include "core/include/network.h"

#include<random>
#include<iostream>

void debug_run() {
    std::default_random_engine generator;

    Network net(2, 3, 1, perceptron_init_hidden(), glorot_weights(generator));

    SignalList test_signals;
    Signal s1 = Signal::from_string("^^__^^__");
    Signal s2 = Signal::from_string("__^^__^^");
    test_signals.add(s1);
    test_signals.add(s2);
    test_signals.equalize_lengths();

    SignalList output = net.forward(test_signals, generator);

    std::cout << "breakpoint" << std::endl;
}


int main(int argc, char** argv) {
    if (argc == 1) {
        debug_run();
        return 0;
    }

    return 0;
}