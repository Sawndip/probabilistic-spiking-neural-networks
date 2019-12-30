#include "core/include/signal.h"
#include "core/include/network.h"
#include "core/include/trainer.h"

#include<iostream>
#include<random>
#include<algorithm>

// Just a function to test callbacks
TrainingProgressTrackAndControlFunction make_train_callback(
    const uint32_t T,
    const SignalList& inputs,
    std::default_random_engine& generator
) {
    return [=,&generator](
        const Network& net,
        const vector<double>& bias_trace_vector, 
        const vector<double>& synapse_trace_vector,
        double mle_log_loss, 
        uint32_t epoch,
        uint32_t t) -> bool {

        if (t == T - 1) {
            std::cout << "Obeserved results after " <<  (epoch + 1) << " epochs of training." << std::endl;

            SignalList f1 = net.forward(inputs, generator);
            std::cout << f1;
        }

        return false;
    };
}

void debug_run() {
    std::default_random_engine generator;
    generator.seed(1337);

    Network net = Network(2, 0, 2,
                          fully_connected_init(), 
                          glorot_weights(generator));

    Signal i1 = Signal::from_string("^^^___");
    Signal i2 = Signal::from_string("___^^^");
    Signal o1 = Signal::from_string("^^^^^^");
    Signal o2 = Signal::from_string("______");

    SignalList inputs;
    inputs.add(i1);
    inputs.add(i2);

    SignalList wanted_outputs;
    wanted_outputs.add(o1);
    wanted_outputs.add(o2);

    FullyObservedOnlineTrainer trainer;

    std::cout << "Observed result with no training." << std::endl;
    SignalList f0 = net.forward(inputs, generator);
    std::cout << f0;

    trainer.train(net, inputs, wanted_outputs, {0.05, 0.0, 10}, 
                 make_train_callback(i1.length(),
                                     inputs, generator));
}

int main(int argc, char** argv) {
    if (argc == 1) {
        debug_run();
    }

    return 0;
}