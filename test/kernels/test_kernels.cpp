#include <core/util.h>

#include <iostream>


int test_ffe(char** argv) {
    const uint32_t window_size = std::stoi(argv[2]);

    double time_constant_1 = std::stod(argv[3]);
    double time_constant_2 = std::stod(argv[4]);

    try {
        auto kernel = exponentially_decaying_feedforward_kernel(
                        window_size, time_constant_1, time_constant_2);
        
        for (double elem: kernel) {
            std::cout << elem << ", ";
        }
    } catch(std::logic_error& e) {
        std::cout << e.what() << std::endl;
        return -1;
    }

    return 0;
}

int test_bfe(char** argv) {
    const uint32_t window_size = std::stoi(argv[2]);

    double time_constant = std::stod(argv[3]);

    try {
        auto kernel = exponentially_decaying_feedback_kernel(
                        window_size, time_constant);
        
        for (double elem: kernel) {
            std::cout << elem << ", ";
        }
    } catch(std::logic_error& e) {
        std::cout << e.what() << std::endl;
        return -1;
    }

    return 0;
}

int main(int argc, char** argv) {
    // debug run when no arguments provided
    if (argc == 1) {
        exponentially_decaying_feedforward_kernel(5, 10.0, 15.0);

        exponentially_decaying_feedback_kernel(1, 3.0);

        return 0; 
    } 

    // Run from ctest when arguments are given
    // One of FFE for feedforward, RE for recursive, RCB for raised cosine basis
    std::string kernel_spec = argv[1];
    if (kernel_spec == "FFE") {
        return test_ffe(argv);
    } else if (kernel_spec == "RE") {
        return test_bfe(argv);
    }
}