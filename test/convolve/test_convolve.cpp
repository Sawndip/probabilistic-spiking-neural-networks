#include <string>
#include <iostream>

#include "core/include/signal.h"
#include "core/include/util.h"


int test(std::string input_signal_spec,
         const uint32_t window_size,
         double time_constant) {

    Signal signal;
    // If needed construct an empty signal
    if (input_signal_spec == "ZERO_LENGTH") {
        signal = Signal(0);
    } else {
        signal = Signal::from_string(input_signal_spec);
    }

    // If needed manualy construct an empty kernel 
    // (so we can test if convolve excepts properly)
    std::vector<double> kernel;

    if (window_size > 0) {
        kernel = exponentially_decaying_feedback_kernel(
            window_size, time_constant
        ); 
    } else {
        kernel = {};
    }

    std::cout.precision(4);

    std::cout << signal << std::endl;

    std::cout << "kernel = ";
    for (auto elem: kernel)
        std::cout << std::fixed << elem << ", ";
    
    std::cout << std::endl;

    if (signal.length() != 0) {
        for (int t = 1; t < signal.length(); t++) {
            double filtered = convolve(kernel, signal.data(), t);

            std::cout << "t = " << t << ", " << std::fixed << filtered << std::endl;
        }
    } else {
        convolve(kernel, signal.data(), 1);
    }

    return 0;
}

int main_test(int argc, char** argv) {
    std::string input_signal_spec = argv[1];

    const uint32_t window_size = std::stoi(argv[2]);

    double time_constant = std::stod(argv[3]);

    try {
        return test(input_signal_spec, window_size, time_constant);
    } catch (std::invalid_argument& err) {
        std::cout << err.what() << std::endl;
        return -1;
    }
}

int main_debug() {
    std::string input_signal_spec = "ZERO_LENGTH";

    const uint32_t window_size = 3;

    double time_constant = 8.0;

    return test(input_signal_spec, window_size, time_constant);
}

int main(int argc, char** argv) {
    if (argc == 1)
        return main_debug();
    
    return main_test(argc, argv);
}