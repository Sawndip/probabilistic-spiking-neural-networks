#ifndef CALLBACKS_H
#define CALLBACKS_H

#include "core/include/network.h"
#include "core/include/util.h"

#include<functional>
#include<string>
#include<iterator>

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
 */ 
TrainingProgressTrackAndControlFunction 
csv_writer(const std::string& output_path, const uint32_t n_neurons);

/*!
 * This function will stop the training process if the L2 gradient magnitude
 * of both gradient vectors is less than the given parameters
 * 
 * \param double epsilon_bias
 * \param double epsilon_synapse
 * 
 * NOTE: The default values are for some dummy test problems in the library.
 * Please change them accordingly for your problem at hand.
 */ 
TrainingProgressTrackAndControlFunction 
stop_on_small_gradients(const double epsilon_bias = 0.45,
                        const double epsilon_synapse = 0.15);

/*!
 * This function will stop the training process if the log loss is greater than
 * the epsilon parameter.
 * 
 * \param double epsilon_loss
 * 
 * NOTE: The default value is for some dummy test problems in the library.
 * Please change it accordingly for your problem at hand.
 */ 
TrainingProgressTrackAndControlFunction
stop_on_acceptable_loss(const double epsilon_loss = -7.0);

// Just a typedef for super long c++ type names
typedef 
std::iterator<std::input_iterator_tag, TrainingProgressTrackAndControlFunction>
TrainingCallbackFunctionsIterator;

/*!
 * Merge some iterable of callback functions into one callback function.
 * The rule is that all calbacks are invoked and the training will stop
 * if at least one callback returns True
 */ 
TrainingProgressTrackAndControlFunction 
merge_callbacks(TrainingCallbackFunctionsIterator begin, 
                TrainingCallbackFunctionsIterator end);

#endif