#include "core/include/util.h"

#include <cmath>
#include <cstdint>

#include <stdexcept>
#include <typeindex>


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

void init_matrix(DoubleMatrix& m, const uint32_t N, const uint32_t M) {
    m.resize(N);
    for(auto& v: m)
        v.resize(M, 0.0);
}

double convolve(const Synapse& syn,
                const DoubleMatrix& matrix,
                const uint32_t t,
                const uint32_t j) {
    const uint32_t K = syn.kernel.size();
    double filtered_trace = 0.0;
    for (uint32_t lag = 0; (lag < K) && (t >= lag); lag++) {
        bool spiked     = matrix[t - lag][j] > 0 ? true : false;
        double kern_val = syn.kernel[lag];

        filtered_trace += spiked * kern_val;
    }
    return filtered_trace;
}