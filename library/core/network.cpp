#include "core/include/network.h"

#include<random>
#include<cmath>
#include<algorithm>

// Everything to everything including self loops
NetworkGeneratorFunction fully_connected_init() {
    return [](const Neuron&, const Neuron&) -> bool { return true; };
}

// Perceptron style without self loops
NetworkGeneratorFunction perceptron_init_hidden() {
    return [](const Neuron& n1, const Neuron& n2) -> bool {
        if (n1.id == n2.id) {
            return false;
        }

        if (n1.type == NeuronType::INPUT && n2.type == NeuronType::HIDDEN) {
            return true;
        } else if (n1.type == NeuronType::HIDDEN && n2.type == NeuronType::OUTPUT) {
            return true;
        }

        return false;
    };
}

NetworkGeneratorFunction perceptron_init_simple() {
    return [](const Neuron& n1, const Neuron& n2) -> bool {
        if (n1.id == n2.id) {
            return false;
        }

        if (n1.type == NeuronType::INPUT && n2.type == NeuronType::OUTPUT) {
            return true;
        }

        return false;
    };
}

NetworkGeneratorFunction random_connections_init(std::default_random_engine generator, 
                                                 const double p) {
    if (p <= 0 || p > 1)
        throw std::invalid_argument("The probability of a synapse forming must be in range (0, 1]");

    return [=, &generator](const Neuron&, const Neuron&) -> bool {
        auto dist = std::uniform_real_distribution<double>(0.0, 1.0);

        double u = dist(generator);

        bool make_synapse = u <= p;

        return make_synapse;
    };
}

KernelInitializerFunction default_exponential_kernels(const uint32_t time_steps,
                                                      const double tau1,
                                                      const double tau2,
                                                      const double tau_m) {
    std::vector<double> feedforward_kern = exponentially_decaying_feedforward_kernel(
        time_steps, tau1, tau2
    );

    std::vector<double> feedback_kern = exponentially_decaying_feedback_kernel(
        time_steps, tau_m
    );

    return [=](const Neuron& n1, const Neuron& n2) {
        if (n1.id == n2.id)
            return feedback_kern;
        else
            return feedforward_kern;
    };
}


WeightInitializerFunction uniform_weights(std::default_random_engine generator,
                                          double a, double b) {
    return [&](const NeuronList&, const Neuron&, const Neuron&) -> double {
        auto dist = std::uniform_real_distribution<double>(a, b);

        return dist(generator);
    };
}

WeightInitializerFunction normal_weights(std::default_random_engine generator, 
                                         double location, double scale) {
    return [&](const NeuronList&, const Neuron&, const Neuron&) -> double {
        auto dist = std::normal_distribution<double>(location, scale);

        return dist(generator);
    };
}

WeightInitializerFunction glorot_weights(std::default_random_engine generator) {
    return [&](const NeuronList&, const Neuron& n1, const Neuron& n2) -> double {
        uint32_t fan_out = n1.successor_neurons.size();
        uint32_t fan_in  = n2.predecessor_neurons.size();

        double scale = 2.0 / ((double)(fan_in + fan_out));

        auto dist = std::normal_distribution<double>(0, scale);

        return dist(generator);
    };
}

void Network::init_neuron_list() {
    if (n_input <= 0)
        throw std::invalid_argument("The number of input neurons must be 1 or greater.");
    if (n_hidden < 0)
        throw std::invalid_argument("The number of hidden neurons must be 0 or greater.");
    if (n_output <= 0)
        throw std::invalid_argument("The number of output neurons must be 1 or greater.");

    uint32_t n_total_neurons = n_input + n_hidden + n_output;

    this->neurons = NeuronList(n_total_neurons);

    for (NeuronId i = 0; i < n_total_neurons; i++) {
        NeuronType type;

        if (i < n_input) {
            type = NeuronType::INPUT;
        } else if (i >= n_input && i < n_input + n_hidden) {
            type = NeuronType::HIDDEN;
        } else {
            type = NeuronType::OUTPUT;
        }
        
        this->neurons[i].type = type;
        this->neurons[i].id   = i;
        this->neurons[i].bias = 0.0;
    }
}

void Network::init_connections(NetworkGeneratorFunction network_gen_func,
                               KernelInitializerFunction kernel_init_func) {    
    const uint32_t n_neurons_total = this->neurons.size();

    this->synapses = SynapseList(n_neurons_total * n_neurons_total);

    for(Neuron& n1: this->neurons) {
        for (Neuron& n2: this->neurons) {
            uint32_t idx = n1.id * n_neurons_total + n2.id;

            if(network_gen_func(n1, n2)) {
                this->synapses[idx].weight = 0.0;
                this->synapses[idx].from   = n1.id;
                this->synapses[idx].to     = n2.id;

                // Assign kernel
                std::vector<double> kernel = kernel_init_func(n1, n2);
                // Check if the kernel is good (contains no NaNs)
                // Raise an exception if it is not good
                bool has_nans = std::any_of(kernel.cbegin(), 
                                            kernel.cend(), 
                                            [](auto& v) { return !std::isfinite(v); });
                if (has_nans) {
                    throw std::invalid_argument("The kernel for a synapse must be composed of finite values.");
                }
                // Copy
                this->synapses[idx].kernel = kernel;

                n1.successor_neurons.push_back(n2.id);
                n2.predecessor_neurons.push_back(n1.id);
            } else {
                // These are the unused synapses. Values initialized to impossible.
                this->synapses[idx].from   = this->neurons.size() + 1e9;
                this->synapses[idx].to     = this->neurons.size() + 1e9;
                this->synapses[idx].weight = std::nan("");
                this->synapses[idx].kernel.assign(
                    this->synapses[idx].kernel.size(), 
                    std::nan(""));
            }
        }
    }

    if (this->synapses.size() == 0)
        throw std::logic_error("There are 0 synapses after the construction procedure");
}

// In the worst case scenario of fully connected with glorot
// this method slows down when caling std::normal_distribution::operator()
// One solution is to develop a special backend thread which will generate standard normals
// The glorot calls can then read from that thread and scale by 2 / (fan_in + fan_out)
// This way the stream of random numbers is already generated before init_weights is called.
void Network::init_weights(WeightInitializerFunction weight_init_func) {
    const uint32_t n_neurons_total = this->neurons.size();

    for(const Neuron& n1: this->neurons) {
        for (const NeuronId& n2_id: n1.successor_neurons) {
            double w = weight_init_func(this->neurons, 
                                        n1, 
                                        this->neurons[n2_id]);

            if (!std::isfinite(w))
                throw std::logic_error("Trying to assign a NaN or infinite weight.");

            uint32_t idx = n1.id * n_neurons_total + n2_id;

            this->synapses[idx].weight = w;
        }
    }
}

Network::Network(uint32_t n_input, 
                 uint32_t n_hidden,
                 uint32_t n_output,
                 NetworkGeneratorFunction network_gen_func,
                 WeightInitializerFunction weight_init_func,
                 KernelInitializerFunction kernel_init_func) {
    
    this->n_input = n_input;
    this->n_hidden = n_hidden;
    this->n_output = n_output;

    this->init_neuron_list();
    this->init_connections(network_gen_func, kernel_init_func);
    this->init_weights(weight_init_func);
}

void Network::check_forward_argument(const SignalList& input) {
    // Check if the input signals and the number of input neurons is equal
    if (input.cdata().size() != this->n_input) {
        throw std::invalid_argument("The number of input signals must match the number of input neurons");
    }

    // Check if the input signals are of equal length
    // Here we use the difference of the sequence T1 T2 T3 ...
    // where Ti is the number of time steps of the i-th signal
    // If T1 = T2 = T3 then the differences will be 0
    // And the sum of differences will be 0 as well
    std::vector<uint32_t> sizes(input.cdata().size());
    std::transform(input.cdata().cbegin(), input.cdata().cend(), sizes.begin(),
    [](const Signal& s) -> uint32_t { return s.length(); });

    std::vector<uint32_t> diff(sizes.size());
    std::adjacent_difference(sizes.begin(), sizes.end(), diff.begin());
    // Due to how adjacent_difference is implemented we must drop the first element.
    diff.erase(diff.begin());
    double total_diff = std::accumulate(diff.begin(), diff.end(), 0);
    if (total_diff != 0) {
        throw std::invalid_argument("All input signals must have the same number of timesteps. Please call equalize_lengths before calling forward.");
    }
}

void __init_matrix(std::vector<std::vector<double>>& matrix, 
                   const uint32_t T, const uint32_t N) {
    matrix.resize(T);
    for(auto& v: matrix) {
        v.resize(N, 0.0);
    }
}

/*!
 * Perform a convolution of previous activations signal with synapse kernel.
 * The parameter t is the current time step in the simulation
 * while pred is the id of the predecessor neuron
 * The parameter matrix is the operation matrix of T rows and N columns.
 */ 
double __convolve(const Synapse& syn,
                  const std::vector<std::vector<double>>& matrix,
                  const uint32_t t,
                  const uint32_t pred) {
    const uint32_t K = syn.kernel.size();
    double filtered_trace = 0.0;
    for (uint32_t lag = 0; (lag < K) && (t >= lag); lag++) {
        bool spiked     = matrix[t - lag][pred] > 0 ? true : false;
        double kern_val = syn.kernel[lag];

        filtered_trace += spiked * kern_val;
    }
    return filtered_trace;
}

SignalList Network::forward(const SignalList& input, 
                            std::default_random_engine& generator) {
    check_forward_argument(input);

    // Do the actual forward pass
    const uint32_t T = input.cdata().begin()->length();
    const uint32_t N = this->neurons.size();

    std::uniform_real_distribution<double> uniform_dist;

    // Operation matrix. T time steps and N neurons.
    // One column is one time step.
    // One row is one neuron.
    // The membrane potentials are stored in this matrix
    // As well as the probabilistically produced spikes
    // after the membrane potential is sigmoided.
    std::vector<std::vector<double>> matrix;

    __init_matrix(matrix, T, N);

    // For all time steps
    for (uint32_t t = 0; t < T; t++) {
        // First load all inputs
        for (NeuronId i = 0; i < this->n_input; i++) {
            matrix[t][i] = input.cdata()[i].cdata()[t];
        }
        
        // Go over all neurons and do the feedforward/feedback
        for (NeuronId i = 0; i < N; i++) {
            // Sum the filtered signals of all predecessor neurons
            // including possible loops
            for (const NeuronId& pred: this->neurons[i].predecessor_neurons) {
                // Find the synapse for this predecessor neuron
                const uint32_t syn_id = pred * N + i;
                const Synapse& syn    = this->synapses[syn_id];

                // Calculate the convolution of 
                // the past activations of predecessor
                // with kernel stored in the synapse
                double filtered_trace = __convolve(syn, matrix, t, pred);

                // Add the weighted contribution found via convolution
                matrix[t][i] += syn.weight * filtered_trace;
            }

            // Add bias of neuron i into the final calculation
            matrix[t][i] += this->neurons[i].bias;
            
            // Calculate the membrane potential of neuron i for time step t
            // by sigmoiding the weighted-sum of filtered traces
            // and probabilistically emit a spike
            double weighted_sum = matrix[t][i];
            double membrane_potential = sigmoid(weighted_sum);
            double u = uniform_dist(generator);
            matrix[t][i] = membrane_potential >= u ? 1 : 0;
        }
    }

    // Construct the signal list from the matrix
    // The output of this function
    SignalList output(this->n_output, T);

    for (uint32_t t = 0; t < T; t++) {
        for (NeuronId i = this->neurons.size() - this->n_output; 
            i < this->neurons.size(); 
            i++) 
        {
            NeuronId j = i - this->n_hidden - this->n_input;
            output.data()[j].data()[t] = matrix[t][i] > 0 ? true : false;
        }
    }

    return output;
}


////////////////////////////////////////////////////////////////////////
// Online SGD training forward pass for one time step. /////////////////
// TODO: Wrap all the vectors and matrices into a struct so the 
// number of parameters to these two functions will be more manageable.
void Network::__train_forward_pass_step(
    const SignalList& example_input,
    const SignalList& wanted_output,

    DoubleMatrix& membrane_potential_matrix,
    DoubleMatrix& saved_filtered_traces,
    DoubleMatrix& saved_membrane_potential_matrix,

    const uint32_t N, const uint32_t t
) { 
    // First load all inputs
    for (NeuronId i = 0; i < this->n_input; i++) {
        membrane_potential_matrix[t][i] = example_input.cdata()[i].cdata()[t];
    }
    
    // Reset the saved filtered traces matrix for a new timestep
    __init_matrix(saved_filtered_traces, N, N);

    // Go over all neurons and do the feedforward/feedback
    for (NeuronId i = 0; i < N; i++) {
        // Sum the filtered signals of all predecessor neurons
        // including possible loops
        for (const NeuronId& pred: this->neurons[i].predecessor_neurons) {
            // Find the synapse for this predecessor neuron
            const uint32_t syn_id = pred * N + i;
            const Synapse& syn    = this->synapses[syn_id];

            // Calculate the convolution of 
            // the past activations of predecessor
            // with kernel stored in the synapse
            double filtered_trace = __convolve(syn, membrane_potential_matrix, t, pred);

            // Add the weighted contribution found via convolution
            membrane_potential_matrix[t][i] += syn.weight * filtered_trace;

            // Save the filtered trace so we can build the gradient from it later
            saved_filtered_traces[pred][i] = filtered_trace;
        }

        // Add bias of neuron i into the final calculation
        membrane_potential_matrix[t][i] += this->neurons[i].bias;
        
        // Calculate the membrane potential of neuron i for time step t
        // by sigmoiding the weighted-sum of filtered traces
        // and probabilistically emit a spike which is stored
        // in the same membrane potential matrix
        double weighted_sum = membrane_potential_matrix[t][i];
        double membrane_potential = sigmoid(weighted_sum);
        
        // As we also need the raw membrane potentials for the gradients
        // We must save them as well.
        saved_membrane_potential_matrix[t][i] = membrane_potential;

        // Set the operating matrix to predetermined ground truth values
        if (this->neurons[i].type == NeuronType::INPUT) {
            membrane_potential_matrix[t][i] = example_input.cdata()[i].cdata()[t];
        } else {
            std::uint32_t j = i - this->n_input;
            membrane_potential_matrix[t][i] = wanted_output.cdata()[j].cdata()[t];
        }
    }
}

void Network::__train_backward_pass_step(
    DoubleMatrix& saved_membrane_potential_matrix,
    DoubleMatrix& membrane_potential_matrix,
    DoubleMatrix& saved_filtered_traces,

    std::vector<double>& bias_trace,
    std::vector<double>& synapse_trace,

    const double et_factor,
    const double learning_rate,

    const uint32_t N, 
    const uint32_t t
) {
    // Terminology from paper and variable names corespondence.
    // s[i, t] = membrane_potential_matrix[t, i] = also x[i, t] 
    // (example_input or wanted_output dependening on i)
    // sigmoid(u[i, t]) = saved_membrane_potential_matrix[t, i]
    // \vec{s[j, i, t-1]} = saved_filtered_traces[j, i]
    for (NeuronId i = 0; i < N; i++) {
        double membrane_potential = saved_membrane_potential_matrix[t][i];
        double spiked             = membrane_potential_matrix[t][i];
        
        // Gradient for the bias term and the
        // time-smoothed version for lowering variance
        double gradient_bias = spiked - membrane_potential;
        double gradient_bias_smoothed;

        // It is not NaN when t > 0
        if (!std::isnan(bias_trace[i])) {
            gradient_bias_smoothed = et_factor * bias_trace[i] + 
                (1 - et_factor) * (gradient_bias);
            bias_trace[i] = gradient_bias_smoothed;
        } else {
            bias_trace[i] = gradient_bias;
            gradient_bias_smoothed = gradient_bias;
        }

        // Update the bias
        this->neurons[i].bias += learning_rate * gradient_bias_smoothed;

        // Gradient for the synapse weights and updates.
        // Note: That we do not treat w[i] as a special case but instead
        // use w[i, i]
        for (const NeuronId& pred: this->neurons[i].predecessor_neurons) {
            // The gradient is the weighted difference between the probability of spiking
            // and whether it spiked or not.
            double sft = saved_filtered_traces[pred][i];
            double gradient_synapse = sft * (spiked - membrane_potential);

            double gradient_synapse_smoothed;

            const uint32_t syn_id = pred * N + i;
            Synapse& syn          = this->synapses[syn_id];         

            // Calculate smoothed version
            // It is not NaN when t > 0
            if (!std::isnan(synapse_trace[syn_id])) {
                synapse_trace[syn_id] = et_factor * synapse_trace[syn_id] +
                    (1 - et_factor) * (gradient_synapse);
                gradient_synapse_smoothed = synapse_trace[syn_id];
            } else {
                synapse_trace[syn_id] = gradient_synapse;
                gradient_synapse_smoothed = gradient_synapse;
            }

            // Update
            syn.weight += learning_rate * gradient_synapse_smoothed;
        }
    }
}

// TODO: Consider moving the training to a new file and class
// This file is getting too big.
// This function implements Algorithm (1) from the paper.
// TODO: Find a way to monitor the weights and changes and to notify so
// the end user will know when to stop training.
// Easiest way would be two matrices, one for gradient history of shape T x N
// One for synapse weight history of shape T x (N x N)
// Then how would one make a decision to stop training? 
// Frobenius norm of these two being small, less then some epsilon?
void Network::train_fully_observed_online(
    const SignalList& example_input,
    const SignalList& wanted_output,
    const double et_factor,
    const double learning_rate,
    const uint32_t n_iterations) {
    
    // TODO: Check that ground truth output is good as well.
    // Has correct number of signals and all of same length as input signals.
    // Also check that the neural network has 0 hidden neurons.
    // Raise exceptions if any problem occurs.
    check_forward_argument(example_input);

    const uint32_t T = example_input.cdata().begin()->length();
    const uint32_t N = this->neurons.size();

    DoubleMatrix membrane_potential_matrix;
    DoubleMatrix saved_membrane_potential_matrix;
    DoubleMatrix saved_filtered_traces;

    __init_matrix(membrane_potential_matrix, T, N);
    __init_matrix(saved_membrane_potential_matrix, T, N);

    // ellegibility traces (averaged gradients over time)
    // At time 0 we must use only the gradient.
    // Instead of checking for time we check for NaNs
    // which is why these arrays are NaN initialized.
    std::vector<double> bias_trace;
    bias_trace.resize(N, std::nan(""));
    std::vector<double> synapse_trace;
    synapse_trace.resize(this->synapses.size(), std::nan(""));

    // For all epochs
    for (uint32_t epoch = 0; epoch < n_iterations; epoch++) {
        // For all time steps
        for (uint32_t t = 0; t < T; t++) {
            // Perform forward pass so we can find the calculated membrane potentials
            // (stored in saved_membrane_potential_matrix), the filtered traces for
            // the current time step and the ground truth values 
            // (stored in membrane_potential_matrix)
            // The reason why ground truth values are stored in membrane_potential_matrix
            // is that we use this matrix for making the forward pass for time t
            // and we need binary spikes/no-spikes from times t-1, t-2 and so on.
            // Compare this to feedforward where we sampled from bernoulli
            // and assigned to the that matrix as well.
            this->__train_forward_pass_step(
                example_input,
                wanted_output,
                membrane_potential_matrix,
                saved_filtered_traces,
                saved_membrane_potential_matrix,
                N, t
            );

            // Forward pass finished for this time step. Calculate the gradients
            // and ellegibility traces and perform in place update.
            this->__train_backward_pass_step(
                saved_membrane_potential_matrix,
                membrane_potential_matrix,
                saved_filtered_traces,
                bias_trace, synapse_trace,
                et_factor, learning_rate,
                N, t
            );
        }
    }
}

// This is also very slow. Of 8 seconds,
// 6 were needed for the printing of the huge 360000 synapse network
ostream& operator<<(ostream& out, const Network& net) {
    out << "NETWORK STATISTICS - NEURONS" << endl;

    out << "INPUT " << net.n_input 
        << " HIDDEN " << net.n_hidden
        << " OUTPUT " << net.n_output 
        << " TOTAL " << net.neurons.size()
        << endl;

    out << "NETWORK STATISTICS - SYNAPSES" << endl;

    out << "TOTAL SYNAPSES " << net.synapses.size() << endl << endl;

    out << "SYNAPSE ADJACENCY LIST (PRESYNAPTIC | LIST POSTSYNAPTIC | TOTAL POST)" << endl;

    std::array<std::string, 3> type_names = {"INPUT", "HIDDEN", "OUTPUT"};

    for (const Neuron& n: net.neurons) {
        out << type_names[n.type] << " " << n.id << " | ";
        for (const NeuronId& n2_id: n.successor_neurons) {
            out << (type_names[net.neurons[n2_id].type]) << " " << n2_id << ", "; 
        }
        out << "| " << n.successor_neurons.size() << endl;
    }

    return out;
}
