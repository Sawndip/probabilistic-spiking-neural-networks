# TODO: Use a variable number of arguments for input
function(do_test target arg_1 arg_2 arg_3 arg_4 result)
    if (NOT TARGET ${target})
        add_executable(${target} ${target}.cpp)

        target_link_libraries(${target} pssn)
    endif()

    set(test_name ${target}_${arg_1}_${arg_2}_${arg_3}_${arg_4})

    message(STATUS "Adding new test ${test_name}"})

    add_test(NAME ${test_name}
             COMMAND ${target} ${arg_1} ${arg_2} ${arg_3} ${arg_4})

    if (NOT ${result} STREQUAL "NO_REGEX")
        set_tests_properties(${test_name}
            PROPERTIES PASS_REGULAR_EXPRESSION ${result}
        )
    endif()
endfunction(do_test)

### CYCLIC GENERATOR ###

do_test(test_cyclic_generator
        5 1 0 "UNUSED"
        "5 \\^\\^\\^\\^\\^")

do_test(test_cyclic_generator
        10 3 1 "UNUSED"
        "10 _\\^__\\^__\\^__")

do_test(test_cyclic_generator
        20 5 0 "UNUSED"
        "20 \\^____\\^____\\^____\\^____")

### MORSE ENCODING ###

# Test for crash on whole alphabet - should not crash
do_test(test_morse_encoding "0123456789abcdefghijklmnopqrstuvwxyz" "10101" "UNU1" "UNU2" "NO_REGEX")

#Test for working on uppercase letters
do_test(test_morse_encoding "fish" "10101" "UNU1" "UNU2" "33 __\\^_\\^_\\^_\\^__\\^_\\^_\\^___\\^_\\^_\\^____\\^_\\^_\\^")
do_test(test_morse_encoding "FISH" "10101" "UNU1" "UNU2" "33 __\\^_\\^_\\^_\\^__\\^_\\^_\\^___\\^_\\^_\\^____\\^_\\^_\\^")

# Test for working on unknown characters (replace unknown by 0 before encoding)
do_test(test_morse_encoding "000" "10101" "UNU1" "UNU2" "30 \\^\\^\\^\\^\\^\\^_\\^_\\^\\^\\^\\^\\^\\^\\^_\\^_\\^\\^\\^\\^\\^\\^\\^_\\^_\\^")
do_test(test_morse_encoding "$!?" "10101" "UNU1" "UNU2" "30 \\^\\^\\^\\^\\^\\^_\\^_\\^\\^\\^\\^\\^\\^\\^_\\^_\\^\\^\\^\\^\\^\\^\\^_\\^_\\^")

# Test for working on empty strings
do_test(test_morse_encoding "E_M_P_T_Y" "10101" "UNU1" "UNU2" "0 ")
