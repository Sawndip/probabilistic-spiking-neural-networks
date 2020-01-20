#include <core/metrics.hpp>

#include <algorithm>
#include <tuple>

namespace core::metrics {
    using Dataset = core::signal::Dataset;

    namespace internal {
        void check_metric_inputs(const Dataset& ground_truth, 
                                 const Dataset& predictions) {
            // ground_truth.check_validity();

            if (ground_truth.cdata().size() != predictions.cdata().size()) {
                throw std::invalid_argument("The number of rows in both ground truth and prediction datasets must match.");
            }
        }
    };

    double hamming_distance(const Dataset& ground_truth, 
                            const Dataset& predictions) {
        internal::check_metric_inputs(ground_truth, predictions);

        double global_sum = 0;

        for (uint32_t i = 0; i < ground_truth.cdata().size(); i++) {
            auto& gt_row = ground_truth.cdata()[i];
            auto& pr_row = predictions.cdata()[i];

            auto& gt_out = std::get<1>(gt_row);
            auto& pr_out = std::get<1>(pr_row);

            double sum = 0;
            double count = 0;

            for (uint32_t k = 0; k < gt_out.number_of_signals(); k++) {
                for (uint32_t j = 0; j < gt_out.time_steps(); j++) {
                    bool gt_entry = gt_out.cdata()[k].cdata()[j];
                    bool pr_entry = pr_out.cdata()[k].cdata()[j];

                    sum += (gt_entry != pr_entry);

                    count++;
                }
            }

            double mean = sum / count;

            global_sum += mean;
        }

        return global_sum / ground_truth.cdata().size();
    }
};