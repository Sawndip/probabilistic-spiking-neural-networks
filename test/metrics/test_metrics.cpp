#include <core/metrics.hpp>
#include <core/signal.hpp>

#include <iostream>
#include <iomanip>

// TODO: Test for invalid_argument crashes if the datasets are not comparable
// TODO: Make some general test interface as all tests follow a general pattern
using namespace core::signal;
using namespace core::metrics;

int test_run(int argc, std::array<std::string, 4> argv) {
    std::string metric_name = argv[0];
    Signal sig1 = Signal::from_string(argv[1]);
    Signal sig2 = Signal::from_string(argv[2]);
    double expected = std::stod(argv[3]);

    SignalList in;in.add(sig1);
    SignalList out;out.add(sig2);

    Dataset ds_1(in, out);
    Dataset ds_2(out, in);

    if (metric_name == "hamming") {
        double hd = hamming_distance(ds_1, ds_2);
        

        std::cout << std::fixed << hd << " " 
                  << std::fixed << expected << " " 
                  << std::fixed << std::abs(expected - hd)
                  << std::endl;
    }

    return 0;
}

void debug_run() {
    std::array<std::string, 4> argv = { "hamming", "^^^^^^", "^^^^^^", "0.0" }; 

    test_run(4, argv);
}

int main(int argc, char** argv) {
    std::cout.precision(4);

    if (argc == 1) {
        debug_run();
        return 0;
    }

    std::array<std::string, 4> argv_cpp11;
    argv_cpp11[0] = argv[1];
    argv_cpp11[1] = argv[2];
    argv_cpp11[2] = argv[3];
    argv_cpp11[3] = argv[4];

    return test_run(argc, argv_cpp11);
}