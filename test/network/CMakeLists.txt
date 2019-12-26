# Test all combinations of network and weight intitializers and make sure none crashes.
# Here we use 2 input, 3 hidden and 1 output neurons.
foreach(init_func RANGE 3)
    foreach (wei_func RANGE 2)
        # Last 0 means we expect no crash
        make_test(
            test_network_cons
            "NO_REGEX"
            2 3 1 ${init_func} ${wei_func} 0
        )
    endforeach()
endforeach()

# Test all combinations of network and weight intitializers and make sure none crashes.
# Here we use 200 input, 300 hidden and 100 output neurons.
# Note that these take around 60 seconds to execute.
# Why so slow? Performance must be upgraded to real time to 
# at least ~1000 neurons in total network
# It is only 2 * 36000 calls of the network gen and weight init functions in worst case.
# Profiler is a good friend here. But some guesses: Use of (for a: b), 
# too many lambda function invocations, too many creations of the dist object?
# Slow map insertion in c++?
foreach(init_func RANGE 3)
    foreach (wei_func RANGE 2)
        # Last 0 means we expect no crash
        make_test(
            test_network_cons
            "NO_REGEX"
            200 300 100 ${init_func} ${wei_func} 0
        )
    endforeach()
endforeach()

# With 20 30 10 neuron distribution.
foreach(init_func RANGE 3)
    foreach (wei_func RANGE 2)
        # Last 0 means we expect no crash
        make_test(
            test_network_cons
            "NO_REGEX"
            20 30 10 ${init_func} ${wei_func} 0
        )
    endforeach()
endforeach()

# Test for crashes when parameters are incorrect.
# Here we have 0 input neurons
make_test(
    test_network_cons
    "NO_REGEX"
    0 0 0 0 0 1
)
# Here we have 0 output neurons
make_test(
    test_network_cons
    "NO_REGEX"
    1 0 0 0 0 1
)
# Here we have -1 hidden neurons
make_test(
    test_network_cons
    "NO_REGEX"
    1 -1 1 0 0 1
)
# Here we have p in random init be 1.5
make_test(
    test_network_cons
    "NO_REGEX"
    1 1 1 3 0 1 1.5
)
# Here we have the case of 0 synapses in the end
make_test(
    test_network_cons
    "NO_REGEX"
    1 1 1 4 0 1
)
# Here we have the case of a weight init function which returns nan
make_test(
    test_network_cons
    "NO_REGEX"
    1 1 1 3 3 1
)