#include "core/include/signal.h"
#include "core/include/network.h"

#include <iostream>
#include <random>

void debug_run() {
    std::default_random_engine generator;
    generator.seed(1337);

    Network net = Network(2, 0, 1,
                          perceptron_init_simple(), 
                          glorot_weights(generator));

    Signal i1 = Signal::from_string("^^^___");
    Signal i2 = Signal::from_string("___^^^");
    Signal o1 = Signal::from_string("^^^^^^");

    SignalList inputs;
    inputs.add(i1);
    inputs.add(i2);

    SignalList wanted_outputs;
    wanted_outputs.add(o1);

    std::cout << "Observed result with no training." << std::endl;
    SignalList f0 = net.forward(inputs, generator);
    std::cout << f0;

    SignalList f1;
    for (std::uint32_t j = 1; j < 10; j++) {
        std::cout << "Obeserved results after " << j << " epochs of training." << std::endl;
        net.train_fully_observed_online(inputs, wanted_outputs, 0.5, 0.02, 1);
        f1 = net.forward(inputs, generator);
        std::cout << f1;
    }
}

int main(int argc, char** argv) {
    if (argc == 1) {
        debug_run();
    }

    return 0;
}