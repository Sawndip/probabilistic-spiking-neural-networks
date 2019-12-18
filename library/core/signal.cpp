#include "core/include/signal.h"

#include <iostream>
#include <algorithm>


Signal::Signal() : signal({0}) {

}

Signal::Signal(const std::uint32_t time_steps) {
    signal.reserve(time_steps);
    signal.assign(time_steps, false);
}

signal_t& Signal::data() {
    return this->signal;
}

const uint32_t Signal::length() const {
    return this->signal.size();
}

void Signal::zero() {
    this->signal.assign(this->signal.size(), false);
}

void Signal::one() {
    this->signal.assign(this->signal.size(), true);
}

void Signal::pad(const uint32_t target_length, const bool value) {
    this->signal.resize(target_length, value);
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

void SignalList::add(const Signal& s) {
    this->signals.push_back(s);
}

void SignalList::equalize_lengths() {
    auto s = std::max_element(signals.begin(), signals.end(),
                              [](const Signal& s1, const Signal& s2) -> bool
                              { return s1.length() < s2.length(); }  );
    
    const uint32_t N = s->length();

    std::for_each(signals.begin(), signals.end(), [N](Signal& s) {
        s.pad(N, false);
    });
}

std::ostream& operator<<(std::ostream& stream, SignalList& signals) {
    stream << signals.data().size() << " signals" << std::endl;

    for (Signal sig: signals.data()) {
        stream << sig << std::endl;
    }

    return stream;
}