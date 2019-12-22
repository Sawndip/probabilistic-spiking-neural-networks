# Test that the code is implemented properly. This requires manual inspection.
make_test(
    test_convolve
"5 \\^___\\^
kernel = -1.0000, -0.8825, -0.7788,[\\r\\n\\t ]*
t = 1, -0.8825[\\r\\n\\t ]*
t = 2, -0.7788[\\r\\n\\t ]*
t = 3, 0.0000[\\r\\n\\t ]*
t = 4, -1.0000"
    # Input signal
    "^___^" 
    # Feedback kernel params
    3 8.0
)

# Test that the convolve method excepts when given a kernel of zero length
make_test(
    test_convolve
    "NO_REGEX"
    "^___^" 
    0 8.0
)

set_tests_properties("test_convolve_^___^_0_8.0" PROPERTIES WILL_FAIL TRUE)

# Test that the convolve method excepts when given a signal of zero length
make_test(
    test_convolve
    "NO_REGEX"
    "ZERO_LENGTH" 
    3 8.0
)

set_tests_properties("test_convolve_ZERO_LENGTH_3_8.0" PROPERTIES WILL_FAIL TRUE)