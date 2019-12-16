#include "core/include/signal.h"

#include <iostream>

Signal::Signal() : signal({0}) {

}

Signal::Signal(const std::uint32_t time_steps) {
    signal.reserve(time_steps);
    signal.assign(time_steps, false);
}

signal_t& Signal::data() {
    return this->signal;
}

const uint32_t Signal::length() {
    return this->signal.size();
}

void Signal::zero() {
    this->data().assign(this->data().size(), false);
}

void Signal::one() {
    this->data().assign(this->data().size(), true);
}

std::ostream& operator<<(std::ostream& stream, Signal& signal) {
    stream << signal.length() << " ";
    for (bool bit: signal.data()) {
        stream << (bit ? '^' : '_');
    }

    return stream;
}

SignalList::SignalList() : signals({}) {
}

SignalList::SignalList(const std::uint32_t n_signals) {
    signals.reserve(n_signals);
}

SignalList::SignalList(const std::uint32_t n_signals, 
                       const std::uint32_t time_steps) {
    signals.reserve(n_signals);
    for (int i = 0; i < n_signals; i++) {
        signals[i] = Signal(time_steps);
    }
}

signal_list_t& SignalList::data() {
    return this->signals;
}

std::ostream& operator<<(std::ostream& stream, SignalList& signals) {
    for (Signal sig: signals.data()) {
        stream << sig << std::endl;
    }

    return stream;
}