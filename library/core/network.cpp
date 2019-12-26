#include "core/include/network.h"

#include <random>

// Everything to everything including self loops
NetworkGeneratorFunction fully_connected_init() {
    return [](Neuron n1, Neuron n2) -> bool { return true; };
}

// Perceptron style without self loops
NetworkGeneratorFunction perceptron_init_hidden() {
    return [](Neuron n1, Neuron n2) -> bool {
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
    return [](Neuron n1, Neuron n2) -> bool {
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
        throw std::logic_error("The probability of a link forming must be in range (0, 1]");

    return [=, &generator](Neuron, Neuron) -> bool {
        auto dist = std::uniform_real_distribution<double>(0.0, 1.0);

        double u = dist(generator);

        bool make_synapse = u <= p;

        return make_synapse;
    };
}

WeightInitializerFunction uniform_weights(std::default_random_engine generator,
                                          double a, double b) {
    return [&](NeuronList, Neuron, Neuron) -> double {
        auto dist = std::uniform_real_distribution<double>(a, b);

        return dist(generator);
    };
}

WeightInitializerFunction normal_weights(std::default_random_engine generator, 
                                         double location, double scale) {
    return [&](NeuronList, Neuron, Neuron) -> double {
        auto dist = std::normal_distribution<double>(location, scale);

        return dist(generator);
    };
}

WeightInitializerFunction glorot_weights(std::default_random_engine generator) {
    return [&](NeuronList, Neuron n1, Neuron n2) -> double {
        uint32_t fan_out = n1.successor_neurons.size();
        uint32_t fan_in  = n2.predecessor_neurons.size();

        double scale = 2.0 / ((double)(fan_in + fan_out));

        auto dist = std::normal_distribution<double>(0, scale);

        return dist(generator);
    };
}

void Network::init_neuron_list() {
    // TODO: Check this is greather than 0, n_input > 0 and n_output > 0 and n_hidden >= 0
    uint32_t n_total_neurons = n_input + n_hidden + n_output;

    this->neurons = NeuronList(n_total_neurons);

    for (int i = 0; i < n_total_neurons; i++) {
        NeuronType type;

        if (i < n_input) {
            type = NeuronType::INPUT;
        } else if (i >= n_input && i < n_input + n_hidden) {
            type = NeuronType::HIDDEN;
        } else {
            type = NeuronType::OUTPUT;
        }

        this->neurons[i] = Neuron();
        this->neurons[i].id = i;
        this->neurons[i].type = type;
        this->neurons[i].bias = 0.0;
    }
}

void Network::init_connections(NetworkGeneratorFunction network_gen_func) {
    for(auto n1: this->neurons) {
        for (auto n2: this->neurons) {
            if(network_gen_func(n1, n2)) {
                Synapse s = {
                    0.0,
                    n1.id,
                    n2.id
                };

                this->synapses[make_tuple(n1.id, n2.id)] = s;

                this->neurons[n1.id].successor_neurons.push_back(n2.id);
                this->neurons[n2.id].predecessor_neurons.push_back(n1.id);
            }
        }
    }
}

void Network::init_weights(WeightInitializerFunction weight_init_func) {
    for(auto n1: this->neurons) {
        for (NeuronId n2_id: n1.successor_neurons) {
            // TODO: Checks that w is not NaN and not +- Inf
            double w = weight_init_func(this->neurons, 
                                        n1, 
                                        this->neurons[n2_id]);

            this->synapses[make_tuple(n1.id, n2_id)].weight = w;
        }
    }
}

Network::Network(uint32_t n_input, 
                 uint32_t n_hidden,
                 uint32_t n_output,
                 NetworkGeneratorFunction network_gen_func,
                 WeightInitializerFunction weight_init_func) {
    
    this->n_input = n_input;
    this->n_hidden = n_hidden;
    this->n_output = n_output;

    this->init_neuron_list();
    this->init_connections(network_gen_func);
    this->init_weights(weight_init_func);
}

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

    for (Neuron n: net.neurons) {
        out << type_names[n.type] << " " << n.id << " | ";
        for (NeuronId n2_id: n.successor_neurons) {
            out << (type_names[net.neurons[n2_id].type]) << " " << n2_id << ", "; 
        }
        out << "| " << n.successor_neurons.size() << endl;
    }

    return out;
}
