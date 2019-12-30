#ifndef TRAINER_H
#define TRAINER_H

#include "core/include/network.h"

using namespace std;


struct TrainingParameters {
    double learning_rate;
    double ellegibility_trace_factor;
    uint32_t epochs;
};


class Trainer {
    public:
        /*!
         * Note: The training procedure modifies the network in place.
         */ 
        void train(
            Network& net,
            const SignalList& input,
            const SignalList& wanted_output,
            const TrainingParameters& params);
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
        vector<double> bias_trace_vector;
        vector<double> synapse_trace_vector;

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
         * TODO: Implement the two extra checks
         */ 
        void check_input_output(
            const Network& net,
            const SignalList& input,
            const SignalList& wanted_output
        ) const;

        void forward_pass_one_time_step(
             const uint32_t t,
             const Network& net,
             const SignalList& input,
             const SignalList& wanted_output);

        double smoothed_bias_gradient(const NeuronId& i, double gradient, double et_factor);
        double smoothed_synapse_gradient(const NeuronId& j, const NeuronId& i, const uint32_t N,
                                         double gradient, double et_factor);

        void update_pass_one_time_step(
            const uint32_t t,
            Network& net,
            const TrainingParameters& params);

    public:
        void train(
            Network& net,
            const SignalList& input,
            const SignalList& wanted_output,
            const TrainingParameters& params);
}; 

#endif