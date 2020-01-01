#include "core/include/signal.h"
#include "core/include/network.h"
#include "core/include/trainer.h"

#include<iostream>
#include<random>
#include<algorithm>

void debug_run() {
    std::default_random_engine generator;
    generator.seed(1337);

    Network net = Network(2, 0, 2,
                          perceptron_init_simple(), 
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

    const uint32_t T = inputs.time_steps();

    auto callback = merge_callbacks({
        on_epoch_end_stats_logger(T),
        on_epoch_end_net_forward(T, inputs, generator),
        // csv_writer("test_fully_observed_trainer.csv", net.total_neurons()),
        stop_on_acceptable_loss(T, -7.0),
        stop_on_small_gradients(T, 0.4, 0.15)
    });

    trainer.train(net, inputs, wanted_outputs, {0.05, 0.5, 40}, callback);
}

int main(int argc, char** argv) {
    if (argc == 1) {
        debug_run();
    }

    return 0;
}