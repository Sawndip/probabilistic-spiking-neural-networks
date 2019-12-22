# Exponential feedforward tests

# Test that it works given acceptable input
make_test(
    test_kernels
    "0, -0.222882, -0.36908, -0.46115, -0.515204, -0.54274, -0.551954, -0.548677, -0.537047, -0.519996, -0.499587, -0.47727, -0.454051,"
    "FFE" 13 3.0 16.0
)

# Test that it crashes when window size is 0
make_test(
    test_kernels
    "NO_REGEX"
    "FFE" 0 0 0
)
set_tests_properties("test_kernels_FFE_0_0_0" 
                     PROPERTIES WILL_FAIL TRUE)

# Test that it crashes when time constant tau1 is zero
make_test(
    test_kernels
    "NO_REGEX"
    "FFE" 1 0 0
)
set_tests_properties("test_kernels_FFE_1_0_0" 
                     PROPERTIES WILL_FAIL TRUE)

# Test that it crashes when time constant tau2 is zero
make_test(
    test_kernels
    "NO_REGEX"
    "FFE" 1 1 0
)
set_tests_properties("test_kernels_FFE_1_1_0" 
                     PROPERTIES WILL_FAIL TRUE)

# Test that it crashes when time constant tau1 is negative
make_test(
    test_kernels
    "NO_REGEX"
    "FFE" 1 -10 0
)
set_tests_properties("test_kernels_FFE_1_-10_0" 
                     PROPERTIES WILL_FAIL TRUE)

# Test that it crashes when time constant tau2 is negative
make_test(
    test_kernels
    "NO_REGEX"
    "FFE" 1 1 -1
)
set_tests_properties("test_kernels_FFE_1_1_-1" 
                     PROPERTIES WILL_FAIL TRUE)

# Test that it outputs 0 when window size is 1 irregardles of time constants
make_test(
    test_kernels
    "0,"
    "FFE" 1 100.0 150.0
)                 

# Exponential feedback tests

# Test that it works given good input arguments
make_test(
    test_kernels 
    "-1, -0.716531, -0.513417, -0.367879, -0.263597," 
    "RE" 5 3.0)

# Test that it crashes for invalid window size
make_test(
    test_kernels 
    "NO_REGEX" 
    "RE" 0 3.0)

set_tests_properties("test_kernels_RE_0_3.0" 
    PROPERTIES WILL_FAIL TRUE)

# Test that it crashes for invalid time constant
make_test(
    test_kernels 
    "NO_REGEX" 
    "RE" 1 -3.0)

set_tests_properties("test_kernels_RE_1_-3.0" 
    PROPERTIES WILL_FAIL TRUE)
