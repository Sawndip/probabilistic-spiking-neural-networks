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
            if(network_gen_func(n1, n2)) {
                uint32_t idx = n1.id * n_neurons_total + n2.id;

                this->synapses[idx].weight = 0.0;
                this->synapses[idx].from   = n1.id;
                this->synapses[idx].to     = n2.id;
                this->synapses[idx].kernel = kernel_init_func(n1, n2);

                n1.successor_neurons.push_back(n2.id);
                n2.predecessor_neurons.push_back(n1.id);
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
    std::vector<uint32_t> sizes;
    std::transform(input.cdata().cbegin(), input.cdata().cend(), sizes.begin(),
    [](const Signal& s) -> uint32_t { return s.length(); });

    std::vector<uint32_t> diff;
    std::adjacent_difference(sizes.begin(), sizes.end(), diff.begin());
    if (std::accumulate(diff.begin(), diff.end(), 0) != 0) {
        throw std::invalid_argument("All input signals must have the same number of timesteps. Please call equalize_lengths before calling forward.");
    }
}


SignalList Network::forward(const SignalList& input) {
    check_forward_argument(input);

    // Do the actual forward pass
    const uint32_t T = input.cdata().begin()->length();

    SignalList output(this->n_output, T);

    for (uint32_t t = 0; t < T; t++) {

    }

    return SignalList();
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
