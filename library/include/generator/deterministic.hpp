#ifndef GENERATOR_DETERMINISTIC_HPP
#define GENERATOR_DETERMINISTIC_HPP

#include <core/signal.hpp>

#include <string>
#include <bitset>

namespace generator::deterministic {
    using Signal = core::signal::Signal;

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

    /*!
    * \brief
    *     Morse encoding of the input string with separator_pattern between characters.
    *     Unknown characters are treated as the '0' character.
    *     Uppercase letters are treated as lowercase.
    * \param Signal& signal Signal - Modified in place
    * \param const std::string& str The string to encode, ideally consisting only of 
    *                                morse encodable characters. It must always be ASCII.
    * \param const uint8_t separator_pattern The between character separator pattern
    *              It should be something like 10101.
    */
    void generate_morse_encode(Signal& signal,
                               const std::string& str,
                               const std::vector<bool>& separator_pattern);

};

#endif