#ifndef UTIL_H
#define UTIL_H

#include "core/include/signal.h"

#include <vector>
#include <cstdint>

/*!
 * \brief The sigmoid function: 1/(1 + exp(-x))
 */
double sigmoid(double x);

/*!
 * \brief The exponentially decaying feedforward kernel (Paper page 6)
 * \param uint32_t window_size Number of time steps to cover
 */ 
std::vector<double>
exponentially_decaying_feedforward_kernel(
    const std::uint32_t window_size,
    const double time_constant_1,
    const double time_constant_2); 

/*!
 * \brief The exponentially decaying feedback kernel (Paper page 7)
 * \param uint32_t window_size Number of time steps to cover
 * \param double time_constant "Duration of the refractory period"
 */ 
std::vector<double>
exponentially_decaying_feedback_kernel(
    const std::uint32_t window_size,
    const double time_constant); 

/*!
 * \brief Convolve the signal with a kernel for one time step t
 */
double convolve(const std::vector<double>& kernel,
                const signal_t& signal, 
                const std::uint32_t t);

#endif