#ifndef CORE_UTIL_HPP
#define CORE_UTIL_HPP

#include <core/signal.hpp>
#include <core/types.hpp>

#include <vector>
#include <cstdint>


namespace core::util {
    using Synapse      = core::types::Synapse;
    using DoubleMatrix = core::types::DoubleMatrix;

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
    * \brief Initialize a matrix with the default element
    * 
    * \param uint32_t N - The first dimension of the matrix (rows)
    * \param uint32_t M - The second dimension of the matrix (columns)
    * \param double v - The value to initialize with
    */ 
    void init_matrix(DoubleMatrix& m, 
                     const uint32_t N,
                     const uint32_t M, 
                     double v = 0);

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

    /*!
    * Calculate a NaN safe L2 norm of a vector.
    * NaNs are treated as 0s
    */ 
    double vector_l2_norm(const std::vector<double>& v);
};

#endif