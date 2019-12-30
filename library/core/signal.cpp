#include "core/include/signal.h"

#include <iostream>
#include <algorithm>


Signal::Signal() : signal({0}) {

}

Signal::Signal(const std::uint32_t time_steps) {
    signal.resize(time_steps);
    signal.assign(time_steps, false);
}

signal_t& Signal::data() {
    return this->signal;
}

const signal_t& Signal::cdata() const {
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

Signal Signal::from_string(std::string spec) {
    Signal signal(spec.size());

    for(int i = 0; i < spec.size(); i++) {
        signal.data()[i] = spec[i] == '^';
    }

    return signal;
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
    signals.resize(n_signals);
    for (int i = 0; i < n_signals; i++) {
        signals[i] = Signal(time_steps);
    }
}

signal_list_t& SignalList::data() {
    return this->signals;
}

const signal_list_t& SignalList::cdata() const {
    return this->signals;
}

const uint32_t SignalList::number_of_signals() const {
    return signals.size();
}

const uint32_t SignalList::time_steps() const {
    if (signals.size() == 0)
        return 0;

    return signals[0].length();
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

void SignalList::concatenate() {
    // Going backwards concatenate all signals into the first one
    for (uint32_t i = signals.size() - 1; i > 0; i--) {
        signal_t& target_sig = signals[i - 1].data();
        signal_t& source_sig = signals[i].data();
        
        target_sig.insert(target_sig.end(), 
                          source_sig.begin(), 
                          source_sig.end());
    }

    // Delete all but the first signal
    signals.resize(1);
}

void SignalList::merge(const SignalList& other) {
    signals.insert(signals.end(), 
                   other.signals.cbegin(), 
                   other.signals.cend());
}

std::ostream& operator<<(std::ostream& stream, SignalList& signals) {
    stream << signals.data().size() << " signals" << std::endl;

    for (Signal sig: signals.data()) {
        stream << sig << std::endl;
    }

    return stream;
}