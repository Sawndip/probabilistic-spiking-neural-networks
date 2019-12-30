#include "core/include/callbacks.h"

#include<fstream>
#include<algorithm>

// TODO: Also write biases and weights to the output, not just the gradients
TrainingProgressTrackAndControlFunction 
csv_writer(const std::string& output_path,
           const uint32_t n_neurons) {
    // Empty the file
    std::fstream file;
    file.open(output_path, std::ios::trunc);
    // Make the header
    file << "# PSNN training progress log #" << std::endl;
    file << "epoch,time,";
    for (uint32_t i = 0; i < n_neurons; i++) {
        file << "bias_gradient_" << i << ",";
    }
    for (uint32_t i = 0; i < n_neurons; i++) {
        for(uint32_t j = 0; j < n_neurons; j++) {
            file << "synapse_gradient_(" << j << ';' << i << "),";
        }
    }
    file << "mle_log_loss" << std::endl; 
    // Close the file as it will be opened again inside the callback
    file.close();

    return [output_path, n_neurons](
        const Network& net, 
        const vector<double>& bias_trace_vector,
        const vector<double>& synapse_trace_vector,
        double mle_log_loss, // unused, to be used later
        uint32_t epoch, 
        uint32_t t
    ) -> bool {
        std::fstream file;
        file.open(output_path, std::ios::app);

        file << epoch << ',' << t << ',';
        for (uint32_t i = 0; i < n_neurons; i++) {
            file << bias_trace_vector[i] << ',';
        }
        for (uint32_t i = 0; i < n_neurons; i++) {
            for(uint32_t j = 0; j < n_neurons; j++) {
                file << synapse_trace_vector[j * n_neurons + i] << ',';
            }
        }
        file << mle_log_loss << std::endl;
        file.close();

        return false;
    };
}

TrainingProgressTrackAndControlFunction 
stop_on_small_gradients(const double epsilon) {
    return [=](
        const Network&, 
        const vector<double>& bias_trace_vector,
        const vector<double>& synapse_trace_vector,
        double,
        uint32_t, 
        uint32_t
    ) -> bool { 
        double grad_norm_bias = vector_l2_norm(bias_trace_vector);
        double grad_norm_syn  = vector_l2_norm(synapse_trace_vector);

        return grad_norm_bias <= epsilon && grad_norm_syn <= epsilon;
    };
}

// TODO: Implement me when internet is available
// Documentation must be consulted for the iterators.
// TrainingProgressTrackAndControlFunction 
// merge(TrainingCallbackFunctionsIterator begin, 
//       TrainingCallbackFunctionsIterator end) {
//     return [&](
//         const Network& net, 
//         const vector<double>& bias_trace_vector,
//         const vector<double>& synapse_trace_vector,
//         double mll,
//         uint32_t epoch, 
//         uint32_t t
//     ) -> bool {
//         bool should_stop = false;

//         auto it = begin;
//         while(it != end) {

//             should_stop = should_stop ||    
//                 it(net, bias_trace_vector, synapse_trace_vector, mll, epoch, t);
//             it++;
//         }

//         return should_stop;
//     };
// }