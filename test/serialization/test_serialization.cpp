#include "core/include/network.h"

#include<random>

int main(int argc, char** argv) {
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
    
    return 0;
}
