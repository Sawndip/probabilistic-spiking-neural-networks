#ifndef NETWORK_H
#define NETWORK_H

#include<vector>
#include<cstdint>
#include<functional>
#include<random>

#include "core/include/types.h"
#include "core/include/util.h"
#include "core/include/signal.h"
#include "core/include/trainer.h"

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
 * TODO: Write docs on the two training methods once they are coded.
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

        // TODO: Decide on a consistent naming convention
        // for the private function names.
        void init_neuron_list();
        void init_connections(NetworkGeneratorFunction synapse_gen_func, 
                              KernelInitializerFunction kernel_init_func);
        void init_weights(WeightInitializerFunction weight_init_func);

        void check_forward_argument(const SignalList& input);

        // Online all visible training
        void __train_forward_pass_step(
                const SignalList& example_input,
                const SignalList& wanted_output,

                DoubleMatrix& membrane_potential_matrix,
                DoubleMatrix& saved_filtered_traces,
                DoubleMatrix& saved_membrane_potential_matrix,

                const uint32_t N, const uint32_t t);

        void __train_backward_pass_step(
            DoubleMatrix& saved_membrane_potential_matrix,
            DoubleMatrix& membrane_potential_matrix,
            DoubleMatrix& saved_filtered_traces,

            std::vector<double>& bias_trace,
            std::vector<double>& synapse_trace,

            const double et_factor,
            const double learning_rate,

            const uint32_t N, 
            const uint32_t t
        );

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
         * Before calling this function ensure equalize length is called
         * 
         * \param SignalList& input - One signal for each input neuron.
         * \param std::default_ranom_engine& generator - The generator from which
         * the algorithm will sample uniformly distributed numbers.
         */
        SignalList forward(const SignalList& input,
                           std::default_random_engine& generator);

        /*!
         * Trains the SNN, with 0 hidden neurons, to generate output signals
         * for input signals. The training is for one data-point.
         * The training happens in an online fashion, i.e. the weights are updated
         * while the network is still doing forward propagation for the later time steps.
         * 
         * \param SignalList& example_input - The input
         * \param SignalList& wanted_output - The wanted output
         * \param double et_factor = 0.5 - The elegibillity trace scaling factor.
         * Larger values will lead to more weight given to previous time gradients
         * \param double learning_rate = 0.01 - The learning rate for the SGD
         * \param const uint32_t n_iterations = 1 - How many times to loop over the 
         * same input. This would be best left 1.
         * 
         * TODO: Decide on how to interface with rest of the world on training progress.
         * Functions are more general, but two matrices for gradient history will do the
         * trick as well.
         */ 
        void train_fully_observed_online(
            const SignalList& example_input,
            const SignalList& wanted_output,
            const double et_factor = 0.5,
            const double learning_rate = 0.01,
            const uint32_t n_iterations = 1);

        friend ostream& operator<<(ostream&, const Network&);
};

#endif 