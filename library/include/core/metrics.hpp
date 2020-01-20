#ifndef CORE_METRICS_HPP
#define CORE_METRICS_HPP

#include <core/signal.hpp>

#include <functional>

namespace core::metrics {
    using Dataset = core::signal::Dataset;

    typedef std::function<double(const Dataset&, const Dataset&)> Metric;

    namespace internal {
        /*!
         * Checks that the lengths of these two are the same.
         * Throws an error if that is not true.
         */ 
        void check_metric_inputs(const Dataset& ground_truth, 
                                 const Dataset& predictions);
    }

    /*!
     * \brief Average of hamming distances between each pair of wanted_output, predicted_output
     * where wanted_output comes from the parameter \param ground_truth while predicted_output comes
     * from the parameter \param predictions
     */ 
    double hamming_distance(const Dataset& ground_truth, 
                            const Dataset& predictions);
};

#endif