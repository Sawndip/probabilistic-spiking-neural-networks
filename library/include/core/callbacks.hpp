#ifndef CORE_CALLBACKS_HPP
#define CORE_CALLBACKS_HPP

#include <core/network.hpp>
#include <core/util.hpp>
#include <core/metrics.hpp>

#include <functional>
#include <string>
#include <initializer_list>
#include <memory>

namespace core::training::callbacks {
    using Network = core::network::Network;
    using SignalList = core::signal::SignalList;
    using Dataset = core::signal::Dataset;
    using Metric = metrics::Metric;

    /*!
    * The parameters in order are:
    * network, bias traces, synapse weight traces, mle_log_loss,
    * epoch, time_step
    */ 
    typedef std::function<bool(const Network&, 
                               const std::vector<double>&,
                               const std::vector<double>&,
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
    * \param const SignalList& input - The input to the neural network
    * \param std::default_random_engine& generator - The generator used for the network forward pass.
    */ 
    TrainingProgressTrackAndControlFunction
    on_epoch_end_net_forward(
        const uint32_t time_steps,
        const SignalList& input_signals,
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
     * The different ways a metric can influence the training procedure.
     * 
     * REPORT means write the value to console
     * LESS_THAN means stop training if the metric value over the whole dataset is 
     * less than the threshold value.
     * GREATER_THAN stops training if the observed metric value over the dataset
     * is above the threshold value.
     * 
     * These three can be mixed using the | operator.
     * 
     * The behaviour is to report when the least significant bit is 1.
     * It will potentially stop the procedure when the second or third bit is 1
     * with the < and > operators accordingly.
     * If both the second and third bits are 1 it will stop when it detects perfect equality
     * (perfect in the bounds of floating point numbers at most)
     */ 
    enum MetricControlType {
        REPORT = 0b001,
        LESS_THAN = 0b010,
        GREATER_THAN = 0b100
    };

    // TODO: Cleanup the trainer and the callback signature
    // So this function won't have so many parameters lying around.
    // Also the other functions and the time_steps parameter is just meh.
    // Much better to have a bool flag like is_epoch_end as part of the callback
    // signature. Or maybe even better an indexing tuple
    // with ids <DatasetRowId, EpochId, TimeId, IsEpochEnd> and as parameters also to provide
    // Dataset ground_truth, Dataset prediction for one row? or all rows?
    // In addition there is no need to call net forward twice just for reporting
    // or metric calculations. The predictions on both train and test
    // sets should be provided by the train method and then metrics and
    // printing of the output signals can be done in the callback methods
    // END OF TODO
    /*!
     * \brief Monitors a metric and potentially stops training if the condition evaluates to true.
     * 
     * \param const Metric& metric - The metric function to evaluate
     * \param const std::string& name - The name to print to stdout
     * \param uint32_t time_steps - The check is done only when epoch ends.
     * We check for epoch end if t == time_steps - 1
     * \param const Dataset& ground_truth - The ground truth dataset
     * \param std::default_random_engine& generator - The generator used for the network forward pass. 
     * \param MetricControlType control_type - The control see \see MetricControlType for docs.
     * \param double thresold_value - The value which determines if we stop or not.
     */ 
    TrainingProgressTrackAndControlFunction
    metric_report_control(const Metric& metric, 
                          const std::string& name,
                          const uint32_t time_steps,
                          const Dataset& ground_truth,
                          std::default_random_engine& generator,
                          uint32_t control_type = MetricControlType::REPORT,
                          double threshold_value = 0.0);

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
};

#endif