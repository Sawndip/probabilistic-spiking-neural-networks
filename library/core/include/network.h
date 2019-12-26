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

    list<NeuronId> predecessor_neurons;
    list<NeuronId> successor_neurons;
};

/*!
 * \brief A list of all neurons in the network.
 * This is the single source of truth for all neurons.
 * Any changes must write directly to this list.
 * To access individual entries the Neuron::id attribute is to be used.
 */ 
typedef vector<Neuron> NeuronList;

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
};

/*!
 * The synapse list is actually a map keyed by a 2-tuple of NeuronId 
 * This is intended to be a single source of truth for all synapses/links in the network.
 * All operations must write only to this class instance.
 */
typedef map<tuple<NeuronId, NeuronId>, Synapse> SynapseList;

/*!
 * A network generator function specifies if a synapse should be formed between
 * a pair of neurons. It is local and stateless in a sense that the only information
 * it is given are two neurons and their properties.
 * 
 * The library provided methods, e.g. fully_connected_init will return a 
 * NetworkGeneratorFunction as their output upon invocation.
 */
typedef function<bool(Neuron, Neuron)> NetworkGeneratorFunction;

/*!
 * \brief Every neuron is connected to every other neuron irregardles of the neuron type
 * 
 * Note that this functions returns a function which does the actual work.
 */
NetworkGeneratorFunction fully_connected_init();

/*!
 * \brief Init the weights in a perceptron fashion with no self-loops and with no 
 * link between output and input. The links are input->output. This function 
 * ignores the hidden neurons.
 * 
 * Note that this functions returns a function which does the actual work.
 */
NetworkGeneratorFunction perceptron_init_simple();

/*!
 * \brief Init the weights in a perceptron fashion with no self-loops and with no 
 * link between output and input. The links are input->hidden->output.
 * 
 *  Note that this functions returns a function which does the actual work.
 */
NetworkGeneratorFunction perceptron_init_hidden();

/*!
 * \brief Create a link between two neurons if a random uniform draw scores below the parameter $p$
 * \param std::default_random_engine The random number generator to use
 * \param double p The probability of link forming. This probaility is indepedent of
 * the previous graph structure and is the same for any two pairs of neurons.
 * 
 * Note that this functions returns a function which does the actual work.
 */
NetworkGeneratorFunction random_connections_init(std::default_random_engine, const double);

/*!
 *  \brief The function which will determine the initial weights for a synapse 
 *  \param NeuronList The list of all neurons. Can be used to acess the structure of the network.
 *  \param Neuron - The presynaptic neuron
 *  \param Neuron - The postsynaptic neuron
 * 
 *  Note: This function is invoked after the network is constructed and it only modifies the weights.
 *  It can not influence the existence of synapses in any form.
 */
typedef function<double(NeuronList, Neuron, Neuron)> WeightInitializerFunction;

/*!
 * \brief Draw the weights from a uniform (a, b) distribution.
 * 
 * \param std::default_random_engine The random number generator to use
 */
WeightInitializerFunction uniform_weights(std::default_random_engine generator,
                                          double a = 0, double b = 1);

/*!
 * \brief Draw the weights from a gaussian (mu=location, sigma=scale) distribution.
 * 
 * \param std::default_random_engine The random number generator to use
 */
WeightInitializerFunction normal_weights(std::default_random_engine generator,
                                         double location = 0, double scale = 1);

/*!
 * Draw the weights from a normal distribution with 0 mean and (2 / (fan_in + fan_out)) std
 * 
 * \param std::default_random_engine The random number generator to use
 */
WeightInitializerFunction glorot_weights(std::default_random_engine generator);

/*!
 * This is a general spiking neural network class for both completely or partialy observed cases.
 * In both cases the forward method is the same while the training procedures differ.
 * TODO: Write docs on the two training methods once they are coded.
 * 
 * The neurons are stored in list in order of INPUT, HIDDEN, OUTPUT
 * The synapses are stored in a map keyed by the pair presynaptic, postsynaptic NeuronId
 * Neither of these are exposed to the public, i.e they are both private.
 * 
 * The public methods are the class constructor, the forward method and the two training methods
 * for a single sample (single SignalList)
 */ 
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
         * \brief Create the spiking neural network.
         * 
         * The first three parameters are the number of input, hidden and output neurons.
         * The 4th parameter is the network generator function
         * The final parameter is the weight initializer function
         */
        Network(uint32_t n_input, 
                uint32_t n_hidden,
                uint32_t n_output,
                NetworkGeneratorFunction network_gen_func,
                WeightInitializerFunction weight_init_func
               );

        /*!
         *
         * 
         * 
         */
        SignalList forward(const SignalList& input);

        friend ostream& operator<<(ostream&, const Network&);
};

#endif 