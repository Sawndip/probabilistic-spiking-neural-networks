#include <core/signal.hpp>
#include <generator/deterministic.hpp>

#include <iostream>

using namespace core::signal;
using namespace generator::deterministic;

void test(const uint32_t n_steps, 
          const uint32_t mod, 
          const uint32_t offset) {
    Signal signal(n_steps);

    generate_cyclic(signal, mod, offset);

    std::cout << signal << std::endl;
}

int main(int argc, char** argv) {
    const uint32_t n_steps = std::stoi(argv[1]);
    const uint32_t mod     = std::stoi(argv[2]);
    const uint32_t offset  = std::stoi(argv[3]);

    test(n_steps, mod, offset);
}