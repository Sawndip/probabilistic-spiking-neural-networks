#include <core/signal.hpp>
#include <generator/deterministic.hpp>

#include <iostream>

void run_test(uint32_t N1, uint32_t N2, uint32_t N3, uint32_t N4) {
    auto mod = 4;
    auto off = 0;

    Signal s1(N1), s2(N2), s3(N3), s4(N4);
    generate_cyclic(s1, mod, off);
    generate_cyclic(s2, mod, off);
    generate_cyclic(s3, mod, off);
    generate_cyclic(s4, mod, off);

    SignalList sl1(2), sl2(2);
    sl1.add(s1);
    sl1.add(s2);
    sl2.add(s3);
    sl2.add(s4);

    std::cout << sl1;

    std::cout << sl2;

    sl1.merge(sl2);

    std::cout << sl1;
}

int main(int argc, char** argv) {
    uint32_t N1  = std::stoi(argv[1]);
    uint32_t N2  = std::stoi(argv[2]);
    uint32_t N3 = std::stoi(argv[3]);
    uint32_t N4 = std::stoi(argv[4]);

    run_test(N1, N2, N3, N4);
}