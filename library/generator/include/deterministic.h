#include "core/include/signal.h"

#ifndef DETERMINISTIC_H
#define DETERMINISTIC_H

/*!
 * \brief 
 *     Cyclic spikes with a fixed offset.
 * 
 * \param Signal& signal Signal already filled with 0s 
 * \param uint32_t mod The number of time steps between spikes. Must be >= 1.
 *    Note that for a single time step of 0 you need this parameter to be 2.
 * \param uint32_t offset The offset until the first spike. Must be >= 0
 */
void generate_cyclic(Signal& signal, 
                     const uint32_t mod,
                     const uint32_t offset);

#endif