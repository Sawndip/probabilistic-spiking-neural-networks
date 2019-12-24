#ifndef NETWORK_H
#define NETWORK_H

#include<vector>
#include<list>
#include<map>
#include<tuple>
#include<cstdint>
#include<functional>
#include<random>

#include "core/include/signal.h"

using namespace std;

/*!
 * 
 */
typedef uint32_t NeuronId;

enum NeuronType {
    INPUT = 0,
    HIDDEN = 1,
    OUTPUT = 2
};

/*!
 * 
 * 
 * 
 */
struct Synapse {
    double weight;

    NeuronId from;
    NeuronId to;
};

typedef map<tuple<NeuronId, NeuronId>, Synapse> SynapseList;

/*!
 *
 * 
 * 
 */
struct Neuron {
    NeuronType type;
    std::uint32_t id;
    double bias;

    list<NeuronId> predecessor_neurons;
    list<NeuronId> successor_neurons;
};

typedef vector<Neuron> NeuronList;

/*!
 *
 * 
 */
typedef function<bool(Neuron, Neuron)> NetworkGeneratorFunction;

/*!
 *
 * 
 */
bool fully_connected_init(Neuron, Neuron);

/*!
 *
 * 
 */
bool perceptron_init_simple(Neuron, Neuron);

/*!
 *
 * 
 */
bool perceptron_init_hidden(Neuron, Neuron);

/*!
 *
 * 
 */
typedef function<double(NeuronList, Neuron, Neuron)> WeightInitializerFunction;

/*!
 *
 * 
 */
WeightInitializerFunction uniform_weights(std::default_random_engine generator,
                                          double a = 0, double b = 1);

/*!
 *
 * 
 */
WeightInitializerFunction normal_weights(std::default_random_engine generator,
                                         double location = 0, double scale = 1);

/*!
 *
 * 
 */
WeightInitializerFunction glorot_weights(std::default_random_engine generator);

class Network {
    private:
        NeuronList neurons;
        SynapseList synapses;

        uint32_t n_input;
        uint32_t n_hidden;
        uint32_t n_output;

        void init_neuron_list();
        void init_connections(NetworkGeneratorFunction synapse_gen_func);
        void init_weights(WeightInitializerFunction weight_init_func);

    public:
        /*!
         *
         * 
         * 
         */
        Network(uint32_t n_input, 
                uint32_t n_hidden,
                uint32_t n_output,
                NetworkGeneratorFunction synapse_gen_func,
                WeightInitializerFunction weight_init_func
               );

        /*!
         *
         * 
         * 
         */
        SignalList forward(const SignalList& input);
};

#endif 