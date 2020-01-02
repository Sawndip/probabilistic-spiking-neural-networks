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

### FORWARD TESTS ###
# The test suite has hardcoded two input neurons
# with two static signals

# Here is a basic test suite that all types of network constructions
# do not crash when called on the forward method
foreach(init_func RANGE 3)
    foreach (wei_func RANGE 2)
        # Last 0 means we expect no crash
        make_test(
            test_network_forward
            "NO_REGEX"
            2 3 1 ${init_func} ${wei_func} 0
        )
    endforeach()
endforeach()

# Same test on a much higher number of hidden and output neurons to check it works as well.
# And does not slow to a crawl...
foreach(init_func RANGE 3)
    foreach (wei_func RANGE 2)
        # Last 0 means we expect no crash
        make_test(
            test_network_forward
            "NO_REGEX"
            2 300 100 ${init_func} ${wei_func} 0
        )
    endforeach()
endforeach()

### FULLY OBSERVED TRAINING TESTS ###
# Test that under normal conditions nothing crashes
# the parameters are
# num input signals, num output signals, time steps of in, time steps of out,
# cyclic generator modulus, cyclic generator offset, 
# learning rate, et factor, epochs,
# input, hidden, output neurons,
# network gen func, weights init func, expect crash
make_test(
    test_fully_observed_train
    "NO_REGEX"
    5 3 100 100 4 7 0.01 0.8 30 5 0 3 0 0 0
)

# Test that an exception is raised when number of hidden > 0
make_test(
    test_fully_observed_train
    "NO_REGEX"
    5 3 100 100 4 7 0.01 0.8 30 5 1 3 0 0 1
)

# Test that an exception is raised when number of output neurons is not same as 
# number of target signals
make_test(
    test_fully_observed_train
    "NO_REGEX"
    5 3 100 100 4 7 0.01 0.8 30 5 0 2 0 0 1
)

# Test that an exception is raised when number of time steps is different for input
# and output
make_test(
    test_fully_observed_train
    "NO_REGEX"
    5 3 100 150 4 7 0.01 0.8 30 5 0 3 0 0 1
)

# Test that exception is raised when the learning rate is not correct
make_test(
    test_fully_observed_train
    "NO_REGEX"
    5 3 100 100 4 7 -0.5 0.8 30 5 0 3 0 0 1
)

make_test(
    test_fully_observed_train
    "NO_REGEX"
    5 3 100 100 4 7 1.5 0.8 30 5 0 3 0 0 1
)

make_test(
    test_fully_observed_train
    "NO_REGEX"
    5 3 100 100 4 7 0.0 0.8 30 5 0 3 0 0 1
)
# Test that exception is raised when et factor is not correct
make_test(
    test_fully_observed_train
    "NO_REGEX"
    5 3 100 100 4 7 0.01 1.2 30 5 0 3 0 0 1
)

make_test(
    test_fully_observed_train
    "NO_REGEX"
    5 3 100 100 4 7 0.01 -1.8 30 5 0 3 0 0 1
)
# Test for an exception when epochs are 0 or negative
make_test(
    test_fully_observed_train
    "NO_REGEX"
    5 3 100 100 4 7 0.01 0.8 0 5 0 3 0 0 1
)

# Test for crash when we invoke with perceptron_init_hidden
# It should crash on basis that no synapses exist in the constructed network
make_test(
    test_fully_observed_train
    "NO_REGEX"
    5 3 100 100 4 7 0.01 0.8 30 5 0 3 2 0 1
)

# Test for various correct network initializations, that no crash is raised
foreach(init_func RANGE 1)
    foreach (wei_func RANGE 2)
        # Last 0 means we expect no crash
        make_test(
            test_fully_observed_train
            "NO_REGEX"
            6 3 100 100 4 7 0.01 0.8 30 6 0 3 ${init_func} ${wei_func} 0
        )
    endforeach()
endforeach()

# Test for performance on a medium-ish network
foreach(init_func RANGE 1)
    foreach (wei_func RANGE 2)
        # Last 0 means we expect no crash
        make_test(
            test_fully_observed_train
            "NO_REGEX"
            20 10 50 50 4 7 0.01 0.8 50 20 0 10 ${init_func} ${wei_func} 0
        )
    endforeach()
endforeach()

# Test for performance on bigger network
# It can get slow depending on number of connections.
# Up to a minute
# foreach(init_func RANGE 1)
#     foreach (wei_func RANGE 2)
#         # Last 0 means we expect no crash
#         make_test(
#             test_fully_observed_train
#             "NO_REGEX"
#             100 50 100 100 4 7 0.01 0.8 30 100 0 50 ${init_func} ${wei_func} 0
#         )
#     endforeach()
# endforeach()