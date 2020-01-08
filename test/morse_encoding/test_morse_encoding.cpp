#include <core/signal.hpp>
#include <generator/deterministic.hpp>

#include <iostream>

void console_test(int argc, char** argv) {
    std::string test_str = argv[1];
    
    if (test_str == "E_M_P_T_Y")
        test_str = "";

    std::vector<bool> separator_pattern;
    for(char c: std::string(argv[2])) {
        separator_pattern.push_back(c == '0' ? false : true);
    }

    Signal signal;

    generate_morse_encode(signal, test_str, separator_pattern);

    std::cout << signal;
}

int main(int argc, char** argv) {
    console_test(argc, argv);
    
    return 0;
}