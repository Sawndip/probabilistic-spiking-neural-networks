#include "core/include/callbacks.h"

#include<fstream>
#include<algorithm>
#include<iostream>
#include<cerrno>

// FIXME: The file is not opened. Investigate why.
// TODO: Also write biases and weights to the output, not just the gradients
TrainingProgressTrackAndControlFunction 
csv_writer(std::string output_path,
           const uint32_t n_neurons) {
    // Empty the file
    std::ofstream file;
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    
    try {
        file.open(output_path, std::ios::trunc);
    } catch (std::system_error& e) {
        std::cerr << e.code().message() << std::endl;

        throw std::runtime_error("Can't open file for CSV logging!");
    }

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
    file.flush();
    // Close the file as it will be opened again inside the callback
    file.close();

    return [output_path, n_neurons](
        const Network& net, 
        const vector<double>& bias_trace_vector,
        const vector<double>& synapse_trace_vector,
        double mle_log_loss,
        uint32_t epoch, 
        uint32_t t
    ) -> bool {
        std::ofstream file_inner;
        file_inner.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        
        try {
            file_inner.open(output_path, std::ios::app);
        } catch (std::system_error& e) {
            std::cerr << e.what() << std::endl;
            std::cerr << e.code().message() << std::endl;

            try {
                throw std::system_error(errno, std::system_category(), "failed to open");
            } catch (std::system_error& e2) {
                std::cerr << e2.what() << std::endl;
            }

            throw std::runtime_error("Can't open file for CSV logging!");
        }
        
        file_inner << epoch << ',' << t << ',';
        for (uint32_t i = 0; i < n_neurons; i++) {
            file_inner << bias_trace_vector[i] << ',';
        }
        for (uint32_t i = 0; i < n_neurons; i++) {
            for(uint32_t j = 0; j < n_neurons; j++) {
                file_inner << synapse_trace_vector[j * n_neurons + i] << ',';
            }
        }
        file_inner << mle_log_loss << std::endl;

        file_inner.flush();
        file_inner.close();

        return false;
    };
}

TrainingProgressTrackAndControlFunction
on_epoch_end_stats_logger(const uint32_t time_steps) {
    return [=](
        const Network& net, 
        const vector<double>& bias_trace_vector,
        const vector<double>& synapse_trace_vector,
        double mle_log_loss,
        uint32_t epoch, 
        uint32_t t
    ) -> bool {
        if (t != time_steps - 1)
            return false;

        std::cout << "epoch: " << epoch << std::endl;
        std::cout << "loss: " << mle_log_loss << std::endl;
        std::cout << "gradient norm bias: " << vector_l2_norm(bias_trace_vector) << std::endl;
        std::cout << "gradient norm synapse: " << vector_l2_norm(synapse_trace_vector) << std::endl;
    
        return false;
    };
}

// DO NOT USE THIS AS IT IS BROKEN
// FIXME: SERIOUS BUG - Problem likely in copy and move constructors of SignalList
// FIXME: The parameters are copied into lambda zero-initialized or random.
// FIXME: Sometimes is correctly copies while most of the time it zero inits or random values.
// FIXME: With the order &generator, inpurs, time_steps it copies time_steps correctly but messes up inputs.
// FIXME: With the order &generator, &inputs, time_steps it still messes up inputs.
// FIXME: shared_ptr did not help as it has 0 count to 0 inside the lambda!
// FIXME: SERIOUS BUG
TrainingProgressTrackAndControlFunction
on_epoch_end_net_forward(
    const uint32_t time_steps,
    std::shared_ptr<SignalList> input_signals,
    std::default_random_engine& generator) {

    // Here it is 2
    std::cout << input_signals.use_count() << std::endl;

    return [&generator, &input_signals, time_steps](
            const Network& net, 
            const vector<double>&,
            const vector<double>&,
            double,
            uint32_t epoch, 
            uint32_t t
        ) -> bool {
            // Here it is 0 - WHY????
            std::cout << input_signals.use_count() << std::endl;

            if (t == time_steps - 1) {
                SignalList out = net.forward(*input_signals, generator);

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
        // Much easier to debug this way, code kept just in case
        // auto it = begin;
        // bool stop = false;
        // while (it != end) {
        //     stop = stop || it->operator()(net, bias_trace_vector, synapse_trace_vector, mll, epoch, t);
        //     it++; }
        // return stop;
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