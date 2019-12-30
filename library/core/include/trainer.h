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
        DoubleMatrix operating_matrix;

        DoubleMatrix saved_membrane_potential_matrix;
        DoubleMatrix saved_filtered_traces_matrix;

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

        void update_pass_one_time_step(
            const uint32_t t,
            Network& net);

    public:
        void train(
            Network& net,
            const SignalList& input,
            const SignalList& wanted_output,
            const TrainingParameters& params);
}; 

#endif