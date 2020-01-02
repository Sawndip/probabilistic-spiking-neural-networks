#ifndef TRAINER_H
#define TRAINER_H

#include "core/include/network.h"
#include "core/include/callbacks.h"


struct TrainingParameters {
    double learning_rate;
    double ellegibility_trace_factor;
    uint32_t epochs;
};


class Trainer {
    public:
        /*!
         * \brief The only public method of this class. It does what it name tells it.
         * 
         * There are two versions, one for no-hidden and one for networks with hidden neurons.
         * The two versions are present in the two subclasses.
         * 
         * Note: The training procedure modifies the network in place.
         * 
         * \param Network& net - The neural network, modified in place.
         * \param const SignalList& input - The input signals.
         * \param const SignalList& wanted_output - The target signals.
         * \param const TrainingParams& params - a struct wrapping the learning rate, e.t. factor and epochs
         * \param const TrainingProgressTrackAndControlFunction callback = nullptr
         * The callback which can be used to display and log the progress as well as to
         * end the training prematurely. Look at callbacks.h for library provided
         * callbacks. Writing custom callbacks is possible as the type of this parameter
         * is std::function. If this parameter is nullptr no callback is involved
         * and the training goes for `param.epochs` epochs.
         */ 
        void train(
            Network& net,
            const SignalList& input,
            const SignalList& wanted_output,
            const TrainingParameters& params,
            TrainingProgressTrackAndControlFunction callback = nullptr);
};

class FullyObservedOnlineTrainer : public Trainer {
    private:
        /// VARIABLES ///
        // This matrix contains the spiking information
        // obtained from ground truth.
        // It is called operating matrix as it is also
        // used in implementing the forward pass.
        // In paper it corresponds to s[i, t]
        // The indexing is time, neuron_id
        DoubleMatrix operating_matrix;

        // This matrix contains the membrane potentials.
        // In paper it corresponds to sigmoid(u[i, t])
        // The indexing is time, neuron_id
        DoubleMatrix saved_membrane_potential_matrix;

        // This matrix contains the filtered traces for 
        // one time step. It is erased and populated
        // for each time step.
        // In paper it corresponds to \vec{s}[j, i, t]
        // for some fixed t and presynaptic j and postsynaptic i
        // The indexing is neuron_id, neuron_id
        DoubleMatrix saved_filtered_traces_matrix;

        // These are the elegibility traces
        // for the smoothed gradients 
        std::vector<double> bias_trace_vector;
        std::vector<double> synapse_trace_vector;

        /// FUNCTIONS ///
        /*!
         * \brief Initialize matrices and vectors to zeros.
         * \param uint32_t T - The number of time steps of the signals.
         * \param uint32_t N - The number of neurons in the network.
         */
        void init_variables(
            const uint32_t T,
            const uint32_t N
        );

        /*!
         * It performs the two checks as in Network::forward
         * and two more additional checks
         * 1. The number of wanted output signals matches the number of output neurons
         * 2. The wanted output signals are of same length as the example input signals
         */ 
        void check_input_output(
            const Network& net,
            const SignalList& input,
            const SignalList& wanted_output
        ) const;

        /*!
         * It checks that learning rate and ellegibility trace factors are between 0 and 1.
         * It also checks that the number of epochs is finite.
         */ 
        void check_training_params(
            const TrainingParameters& params
        ) const;

        /*!
         * Perform a calculation very similar to that of Network::forward
         * The primary difference is that we know the ground truth in this
         * case and thus do not generate spikes randomly.
         * In addition this method keeps more memory then Network::forward.
         * For details look at the comments on variables and in the implementation.
         */ 
        void forward_pass_one_time_step(
             const uint32_t t,
             const Network& net,
             const SignalList& input,
             const SignalList& wanted_output);

        /*!
         * This method calculates a smoothed exp time-averaging of the gradients for bias. 
         */ 
        double smoothed_bias_gradient(const NeuronId& i, 
                                      double gradient, double et_factor);
         /*!
         * This method calculates a smoothed exp time-averaging of the gradients for synapse weights. 
         */ 
        double smoothed_synapse_gradient(const NeuronId& j, const NeuronId& i, 
                                         const uint32_t N,
                                         double gradient, double et_factor);

        /*!
         * This method updates the network's weights and biases after a single time step. 
         * It calculates smoothed gradients and performs the gradient ascent step
         */
        void update_pass_one_time_step(
            const uint32_t t,
            Network& net,
            const TrainingParameters& params);

        /*!
         * Calculate the maximum likelihood loss. 
         * Technically it is a gain as we seek to maximize it.
         */
        double calculate_mll_loss();

    public:
        void train(
            Network& net,
            const SignalList& input,
            const SignalList& wanted_output,
            const TrainingParameters& params,
            TrainingProgressTrackAndControlFunction callback = nullptr);
}; 

#endif