#include "core/include/util.h"

#include <cmath>

double sigmoid(double x) {
    return 1.0 / (1 + std::exp(-x));
}