#ifndef TYPES_H
#define TYPES_H

#include<vector>
#include<functional>
#include<cstdint>

#include<cereal/types/vector.hpp>

/*!
 * A matrix of doubles is used intensively in both forward and
 * training algorithms of the SNN.
 * 
 * The matrix is a vector of vectors.
 */ 
typedef std::vector<std::vector<double>> DoubleMatrix;

/*!
 * The unqiue index into the neuron list for a neuron.
 */
typedef uint32_t NeuronId;

/*!
 * The type of a neuron and the role it plays in the network are defined here as an enum.
 */
enum NeuronType {
    INPUT = 0,
    HIDDEN = 1,
    OUTPUT = 2
};

/*!
 * Each neuron has the following info:
 * 1. NeuronType type - One of INPUT, HIDDEN, OUTPUT
 * 2. NeuronId id - The index in the neuron list where this neuron is stored.
 * 3. double bias - The bias value 
 * 4. list<NeuronId> predecessor_neurons 
 *    The list of all neurons which are presynaptic for this neuron
 * 5. list<NeuronId> successor_neurons
 *    The list of all neurons which are postsynaptic for this neuron
 */
struct Neuron {
    NeuronType type;
    NeuronId id;
    double bias;

    std::vector<NeuronId> predecessor_neurons;
    std::vector<NeuronId> successor_neurons;

    // Cereal serialize method
    template<class Archive>
    void serialize(Archive& archive) {
        archive(type, id, bias, predecessor_neurons, successor_neurons);
    }
};

/*!
 * \brief A list of all neurons in the network.
 * This is the single source of truth for all neurons.
 * Any changes must write directly to this list.
 * To access individual entries the Neuron::id attribute is to be used.
 */ 
typedef std::vector<Neuron> NeuronList;

/*!
 * The Synapse struct contains the following three members:
 * 1. double weight - The weight of the synapse in the feedforward calculation.
 * 2. NeuronId from - The presynaptic neuron
 * 3. NeuronId to   - The postsynaptic neuron
 */
struct Synapse {
    double weight;

    NeuronId from;
    NeuronId to;

    std::vector<double> kernel;
    
    // Cereal serialize method
    template<class Archive>
    void serialize(Archive& archive) {
        archive(weight, from, to, kernel);
    }
};

/*!
 * This is intended to be a single source of truth for all synapses/links in the network.
 * All operations must write only to this class instance.
 */
typedef std::vector<Synapse> SynapseList;

/*!
 * A network generator function specifies if a synapse should be formed between
 * a pair of neurons. It is local and stateless in a sense that the only information
 * it is given are two neurons and their properties.
 * 
 * The library provided methods, e.g. fully_connected_init will return a 
 * NetworkGeneratorFunction as their output upon invocation.
 */
typedef std::function<bool(const Neuron&, const Neuron&)> 
NetworkGeneratorFunction;

typedef std::function<std::vector<double>(const Neuron&, const Neuron&)>
KernelInitializerFunction;

/*!
 *  \brief The function which will determine the initial weights for a synapse 
 *  \param NeuronList The list of all neurons. Can be used to acess the structure of the network.
 *  \param Neuron - The presynaptic neuron
 *  \param Neuron - The postsynaptic neuron
 * 
 *  Note: This function is invoked after the network is constructed and it only modifies the weights.
 *  It can not influence the existence of synapses in any form.
 */
typedef std::function<double(const NeuronList&, const Neuron&, const Neuron&)> 
WeightInitializerFunction;

#endif
