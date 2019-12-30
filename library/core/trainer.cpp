#include "core/include/trainer.h"

void FullyObservedOnlineTrainer::init_variables(
    const uint32_t T,
    const uint32_t N) {

    init_matrix(operating_matrix, T, N);
    init_matrix(saved_membrane_potential_matrix, T, N);
    init_matrix(saved_filtered_traces_matrix, N, N);

    bias_trace_vector.resize(N, std::nan(""));
    synapse_trace_vector.resize(N * N, std::nan(""));
}

void FullyObservedOnlineTrainer::check_input_output(
    const Network& net,
    const SignalList& input,
    const SignalList& wanted_output) const {

    net.check_forward_argument(input);

}

void FullyObservedOnlineTrainer::forward_pass_one_time_step(
    const uint32_t t,
    const Network& net,
    const SignalList& input,
    const SignalList& wanted_output) {

}

void FullyObservedOnlineTrainer::update_pass_one_time_step(
    const uint32_t t,
    Network& net) {
    
}

void FullyObservedOnlineTrainer::train(
    Network& net,
    const SignalList& input,
    const SignalList& wanted_output,
    const TrainingParameters& params) {

    check_input_output(net, input, wanted_output);

    const uint32_t N = net.total_neurons();
    const uint32_t T = input.time_steps();

    init_variables(T, N);
}