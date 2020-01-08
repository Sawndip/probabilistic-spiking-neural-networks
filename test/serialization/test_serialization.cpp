#include "core/include/network.h"

#include<random>

void basic_test() {
    std::default_random_engine generator;
    generator.seed(1337);

    Network net(
        10, 5, 3,
        fully_connected_init(),
        glorot_weights(generator),
        default_exponential_kernels()
    );
    
    auto out_path = "test_serialization.cereal";
    
    net.save(out_path);
    
    Network net2(out_path);
    
    std::cout << "Network 1" << std::endl;
    std::cout << net;
    std::cout << "Network 2" << std::endl;
    std::cout << net2;
    std::cout << "These two should be identical" << std::endl;
}

int test(int argc, char** argv) {
    std::uint32_t n_input  = std::stoi(argv[1]);
    std::uint32_t n_hidden = std::stoi(argv[2]);
    std::uint32_t n_output = std::stoi(argv[3]);

    std::uint32_t ng_func_i = std::stoi(argv[4]);
    std::uint32_t w_func_i  = std::stoi(argv[5]);

    std::uint32_t expect_crash = std::stoi(argv[6]);

    double p = 0.5;
    double wa = 0.0;
    double wb = 1.0;

    int seed = 1337;
    std::default_random_engine generator;
    generator.seed(seed);

    NetworkGeneratorFunction rci;

    try {
        rci = random_connections_init(generator, p);
    } catch (std::invalid_argument&) {
        if (expect_crash)
            return 0;
        
        return -1;    
    }

    std::array<NetworkGeneratorFunction, 4> ng_funcs = {
        fully_connected_init(),
        perceptron_init_simple(),
        perceptron_init_hidden(),
        rci
    };

    std::array<WeightInitializerFunction, 3> wei_funcs = {
        uniform_weights(generator, wa, wb), 
        normal_weights(generator, wa, wb), 
        glorot_weights(generator)
    };

    try {
        Network net = Network(n_input, n_hidden, n_output, 
                              ng_funcs[ng_func_i], wei_funcs[w_func_i]);

        std::string the_path = std::string("/tmp/test_serialization_"); 
        for(uint32_t i = 1; i < 7; i++)
            the_path += std::string(argv[i]) + std::string("_");
        the_path += std::string(".cereal");

        net.save(the_path);

        Network net2(the_path);

        std::system((std::string("du -h ") + the_path).c_str());
    } catch (std::invalid_argument&) {
        if (expect_crash)
            return 0;
        
        return -1;
    } catch (std::logic_error&) {
        if (expect_crash)
            return 0;
        
        return -2;
    }

    return 0;
}

int main(int argc, char** argv) {
    if (argc == 1) {
        basic_test();
        return 0;
    }

    return test(argc, argv);
}
