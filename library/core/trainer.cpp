#include <core/trainer.hpp>

#include <iostream>

void FullyObservedOnlineTrainer::init_variables(
    const uint32_t T,
    const uint32_t N) {

    // Initialize to NaNs so errors in implementation can be caught earlier
    // In addition with such initialization we can implement log-loss calculations
    // for every time step in an easier way.
    init_matrix(operating_matrix, T, N, std::nan(""));
    init_matrix(saved_membrane_potential_matrix, T, N, std::nan(""));
    init_matrix(saved_filtered_traces_matrix, N, N, std::nan(""));

    bias_trace_vector.resize(N, std::nan(""));
    synapse_trace_vector.resize(N * N, std::nan(""));
}

void FullyObservedOnlineTrainer::check_input_output(
    const Network& net,
    const SignalList& input,
    const SignalList& wanted_output) const {

    net.check_forward_argument(input);

    if (wanted_output.number_of_signals() != net.total_outputs())
        throw std::invalid_argument("The ground truth output signals must be same in number as output neurons.");

    if (wanted_output.time_steps() != input.time_steps())
        throw std::invalid_argument("The input and output signals must be of same duration.");
}

void FullyObservedOnlineTrainer::check_training_params(
    const TrainingParameters& params
) const {

    if (params.learning_rate <= 0 || params.learning_rate >= 1)
        throw std::invalid_argument("The learning rate must be in range (0, 1]");

    if (params.ellegibility_trace_factor < 0 || params.ellegibility_trace_factor >= 1)
        throw std::invalid_argument("The elegibilility trace factor must be in range [0, 1]");

    if (!std::isfinite(params.epochs) || params.epochs <= 0)
        throw std::invalid_argument("The number of epochs must be finite and positive.");
}

void FullyObservedOnlineTrainer::forward_pass_one_time_step(
    const uint32_t t,
    const Network& net,
    const SignalList& example_input,
    const SignalList& wanted_output) {
    
    const uint32_t N = net.total_neurons();

    // First load all inputs
    for (NeuronId i = 0; i < net.total_inputs(); i++) {
        operating_matrix[t][i] = example_input.cdata()[i].cdata()[t];
    }

    // And set all outputs to zero before doing forward or backward
    for (NeuronId i = net.total_inputs(); i < net.total_neurons(); i++) {
        operating_matrix[t][i] = 0.0;
    }

    // Reset the saved filtered traces matrix for a new timestep
    init_matrix(saved_filtered_traces_matrix, N, N);

    // Go over all neurons and do the feedforward/feedback
    for (NeuronId i = 0; i < N; i++) {
        // Sum the filtered signals of all predecessor neurons
        // including possible loops
        for (const NeuronId& j: net.cneuron(i).predecessor_neurons) {
            // Find the synapse for this predecessor neuron
            const Synapse& syn = net.csynapse(j, i);

            // Calculate the convolution of 
            // the past activations of predecessor
            // with kernel stored in the synapse
            double filtered_trace = convolve(syn, operating_matrix, t, j);

            // Add the weighted contribution found via convolution
            operating_matrix[t][i] += syn.weight * filtered_trace;

            // Save the filtered trace so we can build the gradient from it later
            saved_filtered_traces_matrix[j][i] = filtered_trace;
        }

        // Add bias of neuron i into the final calculation
        operating_matrix[t][i] += net.cneuron(i).bias;
        
        // Calculate the membrane potential of neuron i for time step t
        // by sigmoiding the weighted-sum of filtered traces
        // and probabilistically emit a spike which is stored
        // in the same membrane potential matrix
        double weighted_sum = operating_matrix[t][i];
        double membrane_potential = sigmoid(weighted_sum);
        
        // As we also need the raw membrane potentials for the gradients
        // We must save them as well.
        saved_membrane_potential_matrix[t][i] = membrane_potential;

        // Set the operating matrix to predetermined ground truth values
        if (net.cneuron(i).type == NeuronType::INPUT) {
            operating_matrix[t][i] = example_input.cdata()[i].cdata()[t];
        } else {
            std::uint32_t j = i - net.total_inputs();
            operating_matrix[t][i] = wanted_output.cdata()[j].cdata()[t];
        }
    }
}

double FullyObservedOnlineTrainer::smoothed_bias_gradient(
    const NeuronId& i,
    double gradient,
    double et_factor) {
    
    double gradient_smoothed;

    // It is not NaN when t > 0
    if (!std::isnan(bias_trace_vector[i])) {
        gradient_smoothed = et_factor * bias_trace_vector[i] + 
            (1 - et_factor) * (gradient);
        bias_trace_vector[i] = gradient_smoothed;
    } else {
        bias_trace_vector[i] = gradient;
        gradient_smoothed = gradient;
    }

    return gradient_smoothed;
}

double FullyObservedOnlineTrainer::smoothed_synapse_gradient(
    const NeuronId& j, const NeuronId& i, const uint32_t N,
    double gradient, double et_factor) {

    double gradient_smoothed;

    const uint32_t syn_id = j * N + i;

    // Calculate smoothed version
    // It is not NaN when t > 0
    if (!std::isnan(synapse_trace_vector[syn_id])) {
        synapse_trace_vector[syn_id] = et_factor * synapse_trace_vector[syn_id] +
            (1 - et_factor) * (gradient);
        gradient_smoothed = synapse_trace_vector[syn_id];
    } else {
        synapse_trace_vector[syn_id] = gradient;
        gradient_smoothed = gradient;
    }

    return gradient_smoothed;
}

void FullyObservedOnlineTrainer::update_pass_one_time_step(
    const uint32_t t,
    Network& net,
    const TrainingParameters& params) {
    
    const uint32_t N = net.total_neurons();
    const double et_factor     = params.ellegibility_trace_factor;
    const double learning_rate = params.learning_rate;

    for (NeuronId i = 0; i < N; i++) {
        Neuron& neuron = net.neuron(i);

        double membrane_potential = saved_membrane_potential_matrix[t][i];
        double spiked             = operating_matrix[t][i];
        
        // Gradient for the bias term and the
        // time-smoothed version for lowering variance
        double gradient_bias = spiked - membrane_potential;
        double gradient_bias_smoothed = smoothed_bias_gradient(i, gradient_bias, et_factor);    

        // Update the bias
        neuron.bias += learning_rate * gradient_bias_smoothed;

        // Gradient for the synapse weights and updates.
        // Note: That we do not treat w[i] as a special case but instead
        // use w[i, i]
        for (const NeuronId& j: neuron.predecessor_neurons) {
            // The gradient is the weighted difference between the probability of spiking
            // and whether it spiked or not.
            double sft = saved_filtered_traces_matrix[j][i];
            double gradient_synapse = sft * (spiked - membrane_potential);

            double gradient_synapse_smoothed = smoothed_synapse_gradient(j, i, N, gradient_synapse, et_factor);

            Synapse& syn = net.synapse(j, i);        

            // Update weight
            syn.weight += learning_rate * gradient_synapse_smoothed;
        }
    }
}

// The log-loss is sum s_i,t * log(u_i,t) + (1 - s_i,t * log(1 - u_i,t))
// The code calculates that formula with some nan checking.
// It provides partial losses in a sense that it can calculate the loss
// after the first timestep, there is no need to wait till end of epoch.
// This is why we need to NaN initialize everything.
double FullyObservedOnlineTrainer::calculate_mll_loss() {
    double log_loss = 0.0;

    for (uint32_t t = 0; t < operating_matrix.size(); t++) {
        for (uint32_t i = 0; i < operating_matrix[0].size(); i++) {
            if (std::isnan(operating_matrix[t][i]))
                return log_loss;

            double p = saved_membrane_potential_matrix[t][i];
            double s = operating_matrix[t][i];

            double a = s * std::log(p) + (1 - s) * std::log(1 - p);

            log_loss += a;
        }
    }

    return log_loss;
}

void FullyObservedOnlineTrainer::train(
    Network& net,
    const SignalList& example_input,
    const SignalList& wanted_output,
    const TrainingParameters& params,
    TrainingProgressTrackAndControlFunction callback) {

    // Perform some checks
    if (net.total_hidden() > 0)
        throw std::logic_error("This algorithm applies only when there are no hidden neurons.");

    check_input_output(net, example_input, wanted_output);

    check_training_params(params);

    const uint32_t N = net.total_neurons();
    const uint32_t T = example_input.time_steps();

    init_variables(T, N);

    bool should_stop = false;

    for (uint32_t epoch = 0; epoch < params.epochs && !should_stop; epoch++) {
        // For all time steps
        for (uint32_t t = 0; t < T && !should_stop; t++) {
            forward_pass_one_time_step(
                t, net, example_input, wanted_output
            );

            update_pass_one_time_step(
                t, net, params
            );

            // Call the callback if there is one
            if (callback != nullptr) {
                double mle_log_loss = calculate_mll_loss();

                should_stop = 
                    callback(net,
                             bias_trace_vector, synapse_trace_vector,
                             mle_log_loss, epoch, t);

                if (should_stop) {
                    std::cout << "Callback commanded stop at epoch " 
                              << epoch << " with time step t = " << t
                              << std::endl;
                    std::cout << "Final loss: " << mle_log_loss << std::endl;
                }
            }
        }
    }

    if (!should_stop)
        std::cout << "Stopping training after all " << params.epochs << " epochs passed." << std::endl;
}