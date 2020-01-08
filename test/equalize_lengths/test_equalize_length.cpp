#include <core/signal.hpp>
#include <generator/deterministic.hpp>

#include <iostream>

using namespace core::signal;
using namespace generator::deterministic;

int main(int argc, char** argv) {
    uint32_t N1  = std::stoi(argv[1]);
    uint32_t N2  = std::stoi(argv[2]);
    uint32_t mod = std::stoi(argv[3]);
    uint32_t off = std::stoi(argv[4]);

    Signal s1(N1), s2(N2);
    generate_cyclic(s1, mod, off);
    generate_cyclic(s2, mod, off);

    SignalList sl(2);
    sl.add(s1);
    sl.add(s2);

    std::cout << sl;

    sl.equalize_lengths();

    std::cout << sl;
}