#include "core/include/callbacks.h"

#include<fstream>
#include<algorithm>
#include<iostream>

// TODO: Also write biases and weights to the output, not just the gradients
TrainingProgressTrackAndControlFunction 
csv_writer(std::string output_path,
           const uint32_t n_neurons,
           const uint32_t time_steps) {
    // Empty the file
    std::ofstream file;
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    
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
    file << "mle_log_loss,epoch_end" << std::endl; 
    file.flush();
    // Close the file as it will be opened again inside the callback
    file.close();

    return [output_path, n_neurons, time_steps](
        const Network&, 
        const vector<double>& bias_trace_vector,
        const vector<double>& synapse_trace_vector,
        double mle_log_loss,
        uint32_t epoch, 
        uint32_t t
    ) -> bool {
        std::ofstream file_inner;
        file_inner.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        
        file_inner.open(output_path, std::ios::app);
    
        file_inner << epoch << ',' << t << ',';
        for (uint32_t i = 0; i < n_neurons; i++) {
            file_inner << bias_trace_vector[i] << ',';
        }
        for (uint32_t i = 0; i < n_neurons; i++) {
            for(uint32_t j = 0; j < n_neurons; j++) {
                file_inner << synapse_trace_vector[j * n_neurons + i] << ',';
            }
        }
        file_inner << mle_log_loss << ","
                   << (t == time_steps - 1 ? 1 : 0)
                   << std::endl;

        file_inner.flush();
        file_inner.close();

        return false;
    };
}

TrainingProgressTrackAndControlFunction
on_epoch_end_stats_logger(const uint32_t time_steps) {
    return [=](
        const Network&, 
        const vector<double>& bias_trace_vector,
        const vector<double>& synapse_trace_vector,
        double mle_log_loss,
        uint32_t epoch, 
        uint32_t t
    ) -> bool {
        // Do not print if not epoch end
        if (t != time_steps - 1)
            return false;

        std::cout << "epoch: " << epoch << std::endl;
        std::cout << "loss: " << mle_log_loss << std::endl;
        std::cout << "gradient norm bias: " << vector_l2_norm(bias_trace_vector) << std::endl;
        std::cout << "gradient norm synapse: " << vector_l2_norm(synapse_trace_vector) << std::endl;
    
        return false;
    };
}

TrainingProgressTrackAndControlFunction
on_epoch_end_net_forward(
    const uint32_t time_steps,
    const SignalList& input_signals,
    std::default_random_engine& generator) {

    return [&generator, &input_signals, time_steps](
            const Network& net, 
            const vector<double>&,
            const vector<double>&,
            double,
            uint32_t epoch, 
            uint32_t t
        ) -> bool {
            if (t == time_steps - 1) {
                SignalList out = net.forward(input_signals, generator);

                std::cout << "Predictions on epoch " << epoch << std::endl;
                std::cout << out;
            }

            return false;
        };
}

TrainingProgressTrackAndControlFunction 
stop_on_small_gradients(    
    const uint32_t time_steps,
    const double epsilon_bias, 
    const double epsilon_synapse) 
{
    return [=](
        const Network&, 
        const vector<double>& bias_trace_vector,
        const vector<double>& synapse_trace_vector,
        double,
        uint32_t, 
        uint32_t t
    ) -> bool { 
        if (t == time_steps - 1) {
            double grad_norm_bias = vector_l2_norm(bias_trace_vector);
            double grad_norm_syn  = vector_l2_norm(synapse_trace_vector);

            return grad_norm_bias <= epsilon_bias && grad_norm_syn <= epsilon_synapse;
        }

        return false;
    };
}

TrainingProgressTrackAndControlFunction
stop_on_acceptable_loss(
    const uint32_t time_steps,
    const double epsilon_loss) 
{
    return [=](
        const Network&, 
        const vector<double>&,
        const vector<double>&,
        double mle_log_loss,
        uint32_t, 
        uint32_t t
    ) -> bool {  
        if (t == time_steps - 1)
            return mle_log_loss >= epsilon_loss;

        return false;
    };
}


TrainingProgressTrackAndControlFunction 
merge_callbacks(const TrainingProgressTrackAndControlFunction* begin,
                const TrainingProgressTrackAndControlFunction* end) {

 if (std::any_of(begin, end, [](const auto& f) { return f == nullptr; })) {
     throw std::invalid_argument("The input collection must not contain nullptr references");
 }

 return [=](
        const Network& net, 
        const vector<double>& bias_trace_vector,
        const vector<double>& synapse_trace_vector,
        double mll,
        uint32_t epoch, 
        uint32_t t
    ) -> bool {
        return std::transform_reduce(begin, end, false, 
            [&](bool b1, bool b2) -> bool { return b1 || b2; },
            [&](const auto& f)    -> bool { return f(net, bias_trace_vector, synapse_trace_vector, 
                                                     mll, epoch, t);  }
        );
    };
}

TrainingProgressTrackAndControlFunction
merge_callbacks(const std::initializer_list<TrainingProgressTrackAndControlFunction>& callbacks) {
    auto begin = callbacks.begin();
    auto end   = callbacks.end();

    return merge_callbacks(begin, end);
}

TrainingProgressTrackAndControlFunction 
merge_callbacks(const std::vector<TrainingProgressTrackAndControlFunction>& callbacks) {
    auto begin = callbacks.data();
    auto end   = callbacks.data() + callbacks.size();

    return merge_callbacks(begin, end);
}