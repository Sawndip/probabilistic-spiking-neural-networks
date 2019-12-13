#include "generator/include/deterministic.h"

void generate_cyclic(Signal& signal, 
                     const uint32_t mod,
                     const uint32_t offset) {
    for (int i = 0; i < signal.length(); i++) {
        if ((i % mod == 0) && (i + offset < signal.length())) {
            signal.data()[i + offset] = true;
        } 
    }
}
