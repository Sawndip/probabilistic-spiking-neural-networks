#ifndef CALLBACKS_H
#define CALLBACKS_H

#include "core/include/network.h"
#include "core/include/util.h"

#include<functional>
#include<string>
#include<initializer_list>
#include<memory>

using namespace std;

/*!
 * The parameters in order are:
 * network, bias traces, synapse weight traces, mle_log_loss,
 * epoch, time_step
 */ 
typedef function<bool(const Network&, 
                      const vector<double>&,
                      const vector<double>&,
                      double,
                      uint32_t, 
                      uint32_t)> 
TrainingProgressTrackAndControlFunction;

/*!
 * Write the full information to a CSV file.
 * For example that CSV file can later be visualized in a jupyter notebook.
 * This function always returns a callback which returns false.
 * 
 * Some values for synapse weights and gradients in the output can be NaN.
 * This indicates the cases when there are no connections.
 * 
 * \param string output_path - Where to write the stats
 * \param uint32_t n_neurons - How many neurons are there in the net, needed for constructing
 * the header
 * \param uint32_t time_steps - How many time steps per epoch, needed for checking for epoch end.
 */ 
TrainingProgressTrackAndControlFunction 
csv_writer(std::string output_path, const uint32_t n_neurons, const uint32_t time_steps);

/*!
 * Show some numbers so we can track how the loss is changing from the command line.
 * 
 * \param uint32_t time_steps - The check is done only when epoch ends.
 * We check for epoch end if t == time_steps - 1
 */ 
TrainingProgressTrackAndControlFunction
on_epoch_end_stats_logger(const uint32_t time_steps);

/*!
 * Peform forward pass on the network after an epoch has ended.
 * Prints the output signals to stdout.
 * 
 * \param uint32_t time_steps - The check is done only when epoch ends.
 * We check for epoch end if t == time_steps - 1 * 
 * \param std::shared_ptr<SignalList> input - The input to the neural network,
 * passed as shared_ptr as the lambda chain can go deep and it is safer this way
 * to pass the shared_ptr instead of references. We try to avoid copies.
 * \param std::default_random_engine& generator - The generator used for the network forward pass.
 */ 
TrainingProgressTrackAndControlFunction
on_epoch_end_net_forward(
    const uint32_t time_steps,
    std::shared_ptr<SignalList> input_signals,
    std::default_random_engine& generator);

/*!
 * This function will stop the training process if the L2 gradient magnitude
 * of both gradient vectors is less than the given parameters
 * 
 * \param uint32_t time_steps - The check is done only when epoch ends.
 * We check for epoch end if t == time_steps - 1
 * \param double epsilon_bias
 * \param double epsilon_synapse
 * 
 * NOTE: The default values are for some dummy test problems in the library.
 * Please change them accordingly for your problem at hand.
 */ 
TrainingProgressTrackAndControlFunction 
stop_on_small_gradients(
    const uint32_t time_steps,
    const double epsilon_bias = 0.45,
    const double epsilon_synapse = 0.15);

/*!
 * This function will stop the training process if the log loss is greater than
 * the epsilon parameter.
 * 
 * \param uint32_t time_steps - The check is done only when epoch ends.
 * We check for epoch end if t == time_steps - 1
 * \param double epsilon_loss
 * 
 * NOTE: The default value is for some dummy test problems in the library.
 * Please change it accordingly for your problem at hand.
 */ 
TrainingProgressTrackAndControlFunction
stop_on_acceptable_loss(
    const uint32_t time_steps,
    const double epsilon_loss = -7.0);

/*!
 * Merge some iterable of callback functions into one callback function.
 * The rule is that all calbacks are invoked and the training will stop
 * if at least one callback returns True
 * 
 * This is the most general version which works with two pointers
 * to start and end of an iterable collection.
 * This should not be used and the two overloads taking a vector or an
 * initializer_list are preferred.
 */ 
TrainingProgressTrackAndControlFunction 
merge_callbacks(const TrainingProgressTrackAndControlFunction* begin,
                const TrainingProgressTrackAndControlFunction* end);

/*!
 * Merge some iterable of callback functions into one callback function.
 * The rule is that all calbacks are invoked and the training will stop
 * if at least one callback returns True
 * 
 * \param initializer_list<...> callbacks - Initializer list of callback functions
 */ 
TrainingProgressTrackAndControlFunction 
merge_callbacks(const std::initializer_list<TrainingProgressTrackAndControlFunction>& callbacks);

/*!
 * Merge some iterable of callback functions into one callback function.
 * The rule is that all calbacks are invoked and the training will stop
 * if at least one callback returns True
 * 
 * \param vector<...> callbacks - Vector of callback functions
 */ 
TrainingProgressTrackAndControlFunction 
merge_callbacks(const std::vector<TrainingProgressTrackAndControlFunction>& callbacks);

#endif