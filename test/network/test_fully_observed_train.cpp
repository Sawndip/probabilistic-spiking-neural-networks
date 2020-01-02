#include "core/include/signal.h"
#include "core/include/network.h"
#include "core/include/trainer.h"
#include "generator/include/deterministic.h"

#include<iostream>
#include<random>
#include<algorithm>

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

    SignalList wanted_outputs;
    wanted_outputs.add(o1);
    wanted_outputs.add(o2);

    std::shared_ptr<SignalList> inputs_ptr(new SignalList());
    inputs_ptr->add(i1);
    inputs_ptr->add(i2);

    FullyObservedOnlineTrainer trainer;

    std::cout << "Observed result with no training." << std::endl;
    SignalList f0 = net.forward(*inputs_ptr, generator);
    std::cout << f0;

    const uint32_t T = inputs_ptr->time_steps();

    std::string output_csv_path("test_fully_observed_trainer.csv");

    std::initializer_list<TrainingProgressTrackAndControlFunction>
    callbacks = {
        on_epoch_end_stats_logger(T),
        on_epoch_end_net_forward(T, inputs_ptr, generator),
        csv_writer(output_csv_path, net.total_neurons(), T),
        stop_on_acceptable_loss(T, -7.0),
        stop_on_small_gradients(T, 0.4, 0.15)
    };
    
    auto callback = merge_callbacks(callbacks);

    trainer.train(net, *inputs_ptr, wanted_outputs, {0.05, 0.5, 40}, callback);
}

void make_input_output(SignalList* out, SignalList* in,
                       uint32_t in_sigs, uint32_t out_sigs,
                       uint32_t time_1,
                       uint32_t time_2,
                       uint32_t mod, 
                       uint32_t off) {
    for (uint32_t i = 0; i < in_sigs; i++) {
        Signal s(time_1);
        generate_cyclic(s, mod, off);
        in->add(s);
    }

    for (uint32_t j = 0; j < out_sigs; j++) {
        Signal s2(time_2);
        generate_cyclic(s2, mod, off);
        out->add(s2);
    } 

    std::cout << " inside the make method " << out->number_of_signals() << " output signals\n";

    std::cout << " inside the make method " << in->number_of_signals() << " input signals\n";
}

// Note: Interestingly enough - glorot weights are worst offender here as well,
// them slowing down the construction of the net

// Note callbacks are not tested
// as that would make this test even more complicated
// It would be good to have a test for callbacks as well
// primarily for their performance when n_input and n_output become large.
// and network is fully connected.
// Still due to time issues as of 02.01.2020 this feature remains to be 
// developed later
int test_run(int argc, char** argv) {
    SignalList in;
    SignalList out;

    make_input_output(&in, &out,
                      std::stoi(argv[1]), 
                      std::stoi(argv[2]),
                      std::stoi(argv[3]), 
                      std::stoi(argv[4]),
                      std::stoi(argv[5]), 
                      std::stoi(argv[6]));
    
    std::cout << "After make_input_output input signals are " << in.number_of_signals() << std::endl;
    std::cout << "After make_input_output output signals are " << out.number_of_signals() << std::endl;

    // Ok weird bug where everything is good inside the func
    // but outside in and out appear swapped???,
    // here is a quick fix
    // FIXME: Investigate more why this happens
    std::swap(in, out);

    TrainingParameters params;
    params.learning_rate = std::stod(argv[7]);
    params.ellegibility_trace_factor = std::stod(argv[8]);
    params.epochs = std::stoi(argv[9]);

    std::uint32_t n_input  = std::stoi(argv[10]);
    std::uint32_t n_hidden = std::stoi(argv[11]);
    std::uint32_t n_output = std::stoi(argv[12]);

    std::uint32_t ng_func_i = std::stoi(argv[13]);
    std::uint32_t w_func_i  = std::stoi(argv[14]);

    std::uint32_t expect_crash = std::stoi(argv[15]);

    std::array<NetworkGeneratorFunction, 3> ng_funcs = {
        fully_connected_init(),
        perceptron_init_simple(),
        perceptron_init_hidden()
    };

    double wa = 0.0;
    double wb = 1.0;
    int seed = 1337; 
    std::default_random_engine generator;
    generator.seed(seed);

    std::array<WeightInitializerFunction, 3> wei_funcs = {
        uniform_weights(generator, wa, wb), 
        normal_weights(generator, wa, wb), 
        glorot_weights(generator)
    };

    FullyObservedOnlineTrainer trainer;

    in.equalize_lengths();
    out.equalize_lengths();

    try {
        Network net = Network(n_input, n_hidden, n_output, 
                              ng_funcs[ng_func_i], wei_funcs[w_func_i]);

        trainer.train(net, in, out, params, nullptr);
    } catch(std::exception& ex) {
        std::cout << ex.what() << std::endl;

        if (expect_crash)
            return 0;
        else
            return -1;
    }

    if (expect_crash)
        return -1;

    return 0;
}

int main(int argc, char** argv) {
    if (argc == 1) {
        debug_run();
    }

    return test_run(argc, argv);
}