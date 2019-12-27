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
        const uint32_t length() const;

        /*!
         * \brief Make all bits zero
         */
        void zero();

        /*!
         * \brief Make all bits one
         */
        void one();

        /*!
         * \brief Pad to the right until meeting the target number of time steps.
         * 
         * \param uint32_t target_length Number of desired timesteps
         * \param bool value The value to pad with, default is 0
         */
        void pad(const uint32_t target_length, const bool value = false);

        /*!
         * \brief Construct a signal from string treating ^ as 1, _ as 0 
         * and everything else as 0 as well
         */
        static Signal from_string(std::string spec);

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

        const signal_list_t& cdata() const;

        /*!
         * \brief Add a new signal to the end of this list.
         */ 
        void add(const Signal& s);

        /*!
         * \brief Pad all signals so they will have same length as the longest signal.
         * This method will modify the instance in place.
         */
        void equalize_lengths();

        /*!
         * \brief Concatenate all signals into one.
         * At the end this SignalList will contain only one signal.
         * This method will modify the instance in place.
         */ 
        void concatenate();

        /*!
         * \brief Merge two signal lists into one
         * 
         * \param SignalList& other The source of the signals to be merged
         */
        void merge(const SignalList& other);

        SignalList();
        SignalList(const uint32_t n_signals);
        SignalList(const uint32_t n_signals, const uint32_t time_steps);

        friend std::ostream& operator<<(std::ostream&, SignalList&); 
};

#endif