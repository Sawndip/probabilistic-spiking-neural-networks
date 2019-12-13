#ifndef SIGNAL_H
#define SIGNAL_H

#include <vector>
#include <ostream>

typedef std::vector<bool> signal_t;

/*! 
 * \brief The binary signal used for input, output and intermediate processing in the network.
 * 
 * The signal is implemented as a vector in the background so it can be dynamicaly resized
 * if necessary.
 */
class Signal {
    private:
        signal_t signal;
    public:
        /*!
         * \brief Access the internal representation of this Signal.
         * 
         * Recomended to be used only in the library and not for public use.
         */
        signal_t& data();

        /*!
         * \brief The number of time steps this signal lasts
         */
        const uint32_t length();

        Signal();
        Signal(const std::uint32_t time_steps);

        friend std::ostream& operator<<(std::ostream&, Signal&); 
};

typedef std::vector<Signal> signal_list_t;

/*! 
 * \brief A list of binary signals internaly implemented as vector.
 */
class SignalList {
    private:
        signal_list_t signals;
    public:
        /*!
         * \brief Access the signals of this list. 
         */
        signal_list_t& data();

        SignalList();
        SignalList(const uint32_t n_signals);
        SignalList(const uint32_t n_signals, const uint32_t time_steps);

        friend std::ostream& operator<<(std::ostream&, SignalList&); 
};

#endif