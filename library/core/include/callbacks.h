#ifndef CALLBACKS_H
#define CALLBACKS_H

#include "core/include/network.h"

#include<functional>
#include<string>

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
 */ 
TrainingProgressTrackAndControlFunction 
csv_writer(const std::string& output_path);

/*!
 * This function will stop the training process if the L2 gradient magnitude
 * of both gradient vectors is less than the given parameter epsilon
 * 
 * \param double epsilon
 */ 
TrainingProgressTrackAndControlFunction 
stop_on_small_gradients(const double epsilon = 1e-6);

// Just a typedef for super long c++ type names
typedef 
iterator<forward_iterator_tag, TrainingProgressTrackAndControlFunction>
TrainingCallbackFunctionsIterator;

/*!
 * Merge some iterable of callback functions into one callback function.
 * The rule is that all calbacks are invoked and the training will stop
 * if at least one callback returns True
 */ 
TrainingProgressTrackAndControlFunction 
merge(TrainingCallbackFunctionsIterator begin, 
      TrainingCallbackFunctionsIterator end);

#endif