#include "core/include/util.h"

#include <cmath>
#include <cstdint>

#include <stdexcept>


double sigmoid(double x) {
    return 1.0 / (1 + std::exp(-x));
}

std::vector<double> 
exponentially_decaying_feedforward_kernel(
    const std::uint32_t window_size,
    const double time_constant_1,
    const double time_constant_2) {

    // TODO: Research if some more checks make sense, e.g. tau1 < tau2, window_size < tau2 - tau1?
    if (window_size == 0)
        throw std::invalid_argument("Window size must be positive.");
    if (time_constant_1 <= 0)
        throw std::invalid_argument("The time constant tau1 must be positive");
    if (time_constant_2 <= 0)
        throw std::invalid_argument("The time constant tau2 must be positive");

    std::vector<double> kernel(window_size, 0);

    for (std::uint32_t t = 0; t < window_size; t++) {
        double t1 = -((double)t / time_constant_1);
        double t2 = -((double)t / time_constant_2);

        double v = std::exp(t1) - std::exp(t2);

        kernel[t] = v;
    }

    return kernel;
}

std::vector<double>
exponentially_decaying_feedback_kernel(
    const std::uint32_t window_size,
    const double time_constant) {
    
    if (window_size == 0)
        throw std::invalid_argument("Window size must be positive.");
    if (time_constant <= 0)
        throw std::invalid_argument("The time constant tau_m must be positive");

     std::vector<double> kernel(window_size, 0);

    for (std::uint32_t t = 0; t < window_size; t++) {
        double v = -std::exp((-(double)t) / time_constant);

        kernel[t] = v;
    }

    return kernel;
}

double convolve(const std::vector<double>& kernel,
                const signal_t& signal, 
                const std::uint32_t t) {
    // Input checks
    if (t < 1)
        throw std::invalid_argument("The time step must be 1 or greater.");
    if(kernel.size() < 1)
        throw std::invalid_argument("The kernel must have at least one time step");
    if (signal.size() < 1)
        throw std::invalid_argument("The signal must have at least one time step");

    // Convolution code
    double result = 0;

    for (int i = 0; i < kernel.size() && (int)t - i >= 0; i++) {
        result += kernel[i] * signal[t - i];
    }

    return result;
}