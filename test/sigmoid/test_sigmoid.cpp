#include <core/util.hpp>

#include <iostream>
#include <iomanip>

using namespace core::util;


int main(int argc, char** argv) {
    std::cout << "x, sigmoid(x)" << std::endl;

    for (double x = -5.0; x < 5.0; x += 0.25) {
        double y = sigmoid(x);

        std::cout.precision(12);
        std::cout << "(" << std::fixed << x << ',' << std::fixed << y << ")" << std::endl;
    }

    std::cout << std::endl;

    std::cout << "x, sigmoid(x), sigmoid(-x), sigmoid(x) + sigmoid(-x)" << std::endl;

    for (double x = 5; x >= 0; x -= 0.25) {
        std::cout << std::fixed << x << ", ";
        std::cout << std::fixed << sigmoid(x) << ", ";
        std::cout << std::fixed << sigmoid(-x) << ", ";
        std::cout << std::fixed << sigmoid(x) + sigmoid(-x) << std::endl;
    }
}