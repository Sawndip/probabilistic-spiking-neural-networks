#include "core/include/network.h"

#include <random>

// Everything to everything including self loops
bool fully_connected_init(Neuron n1, Neuron n2) {
    return true;
}

// Perceptron style without self loops
bool perceptron_init_hidden(Neuron n1, Neuron n2) {
    if (n1.id == n2.id) {
        return false;
    }

    if (n1.type == NeuronType::INPUT && n2.type == NeuronType::HIDDEN) {
        return true;
    } else if (n1.type == NeuronType::HIDDEN && n2.type == NeuronType::OUTPUT) {
        return true;
    }

    return false;
}

bool perceptron_init_simple(Neuron n1, Neuron n2) {
    if (n1.id == n2.id) {
        return false;
    }

    if (n1.type == NeuronType::INPUT && n2.type == NeuronType::OUTPUT) {
        return true;
    }

    return false;
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

void Network::init_connections(NetworkGeneratorFunction synapse_gen_func) {
    for(auto n1: this->neurons) {
        for (auto n2: this->neurons) {
            if(synapse_gen_func(n1, n2)) {
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
                 NetworkGeneratorFunction synapse_gen_func,
                 WeightInitializerFunction weight_init_func) {
    
    this->n_input = n_input;
    this->n_hidden = n_hidden;
    this->n_output = n_output;

    this->init_neuron_list();
    this->init_connections(synapse_gen_func);
    this->init_weights(weight_init_func);
}