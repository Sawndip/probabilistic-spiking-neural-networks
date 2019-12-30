#ifndef NETWORK_H
#define NETWORK_H

#include<vector>
#include<cstdint>
#include<functional>
#include<random>

#include "core/include/types.h"
#include "core/include/util.h"
#include "core/include/signal.h"

using namespace std;

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

// Default values put up randomly. TODO: Research on sane defaults
KernelInitializerFunction default_exponential_kernels(const uint32_t time_steps = 7,
                                                      const double tau1 = 3,
                                                      const double tau2 = 11,
                                                      const double tau_m = 5);

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
 * To train \see the class Trainer
 * 
 * The neurons are stored in vector in order of INPUT, HIDDEN, OUTPUT
 * The synapses are stored in a vector keyed by 
 *  presynaptic * n_total_neurons + postsynaptic NeuronId
 * Neither of these are exposed to the public, i.e they are both private.
 * In both cases the values are stored without use of pointers and without
 * manual new/delete type of memory management.
 */ 
class Network {
    private:
        NeuronList neurons;
        SynapseList synapses;

        uint32_t n_input;
        uint32_t n_hidden;
        uint32_t n_output;

        void init_neuron_list();
        void init_connections(NetworkGeneratorFunction synapse_gen_func, 
                              KernelInitializerFunction kernel_init_func);
        void init_weights(WeightInitializerFunction weight_init_func);

    public:
        /*!
         * \brief Create the spiking neural network.
         * 
         * The first three parameters are the number of input, hidden and output neurons.
         * The 4th parameter is the network generator function
         * The 5th parameter is the weight initializer function.
         * The 6th parameter is the synapse kernel initializer function.
         */
        Network(uint32_t n_input, 
                uint32_t n_hidden,
                uint32_t n_output,
                NetworkGeneratorFunction network_gen_func,
                WeightInitializerFunction weight_init_func,
                KernelInitializerFunction kernel_init_func = default_exponential_kernels()
               );

        /*!
         * \brief Accessor method for a single Neuron.
         * Going out of bounds will cause a segmentation fault.
         */ 
        Neuron& neuron(const NeuronId id);
        /*!
         * \brief Const accessor method for accessing a neuron.
         * 
         * \see Network::neuron
         */ 
        const Neuron& cneuron(const NeuronId id) const;

        /*!
         * \brief Accessor method for a synapse starting in neuron j and ending in i.
         * 
         * \param NeuronId j - The presynaptic neuron id
         * \param NeuronId i - The postsynaptic neuron id
         */ 
        Synapse& synapse(const NeuronId j, const NeuronId i);

        /*!
         * \brief Const accessor method for synapses.
         * \see Network::synapse
         */ 
        const Synapse& csynapse(const NeuronId j, const NeuronId i) const;

        /*!
         * Return the number of neurons in this network.
         * The other 3 are very similar and self explanatory.
         */ 
        const uint32_t total_neurons() const;
        const uint32_t total_inputs() const;
        const uint32_t total_hidden() const;
        const uint32_t total_outputs() const;

        /*!
         * Performs two checks
         * 1. The number of provided signals matches the number of input neurons.
         * 2. The length of all signals is equal to a single value, i.e. all signals
         * have the same length.
         */ 
        void check_forward_argument(const SignalList& input) const;

        /*!
         * Before calling this function ensure equalize length is called
         * 
         * \param SignalList& input - One signal for each input neuron.
         * \param std::default_ranom_engine& generator - The generator from which
         * the algorithm will sample uniformly distributed numbers.
         */
        SignalList forward(const SignalList& input,
                           std::default_random_engine& generator);

        friend ostream& operator<<(ostream&, const Network&);
};

#endif 