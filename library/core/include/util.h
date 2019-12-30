#ifndef UTIL_H
#define UTIL_H

#include "core/include/signal.h"
#include "core/include/types.h"

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
 * \brief Initialize a matrix with the default element (usually 0)
 * 
 * \param uint32_t N - The first dimension of the matrix (rows)
 * \param uint32_t M - The second dimension of the matrix (columns)
 */ 
void init_matrix(DoubleMatrix& m, const uint32_t N, const uint32_t M);

/*!
 * Perform a convolution of previous activations signal with synapse kernel.
 * The parameter t is the current time step in the simulation
 * while j is the id of the predecessor (presynaptic) neuron
 * The parameter matrix is the operation matrix of T rows and N columns.
 */ 
double convolve(const Synapse& syn,
                const DoubleMatrix& matrix,
                const uint32_t t,
                const uint32_t j);

/*!
 * \brief Outputs a vector to stdout.
 * \param vector<double> v - The vector to print
 * \param string name - The name of the vector
 */ 
void print_vector(std::vector<double> v, std::string name);

double vector_l2_norm(const std::vector<double>& v);

#endif