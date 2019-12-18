function(make_test)
    if (${ARGC} LESS 2)
        message(STATUS "make_test ERROR: You must provide at least two arguments")
        message(STATUS "name of target and regex pattern to match on or NO_REGEX")
        return()
    endif()

    set(target ${ARGV0})
    set(result ${ARGV1})
    list(SUBLIST ARGV 2 -1 other_args)

    # If not built already, build the test executable
    if (NOT TARGET ${target})
        add_executable(${target} ${target}.cpp)

        target_link_libraries(${target} pssn)
    endif()

    # Add a new test
    string(REPLACE ";" "_" test_name_args "${other_args}")
    set(test_name ${target}_${test_name_args})
    message(STATUS "Adding new test ${test_name}")
    add_test(NAME ${test_name}
             COMMAND ${target} ${other_args})

    # If we want to check if the STDOUT of the test satisfies a certain regex
    # we add it here
    if (NOT ${result} STREQUAL "NO_REGEX")
        set_tests_properties(${test_name}
            PROPERTIES PASS_REGULAR_EXPRESSION ${result}
        )
    endif()
endfunction()

### CYCLIC GENERATOR ###

make_test(
    test_cyclic_generator
    "5 \\^\\^\\^\\^\\^"    
    5 1 0)

make_test(
    test_cyclic_generator
    "10 _\\^__\\^__\\^__"
    10 3 1)

make_test(
    test_cyclic_generator
    "20 \\^____\\^____\\^____\\^____"
    20 5 0)

### MORSE ENCODING ###

# Test for crash on whole alphabet - should not crash
make_test(
    test_morse_encoding 
    "NO_REGEX"
    "0123456789abcdefghijklmnopqrstuvwxyz" "10101")

#Test for working on uppercase letters
make_test(
    test_morse_encoding 
    "33 __\\^_\\^_\\^_\\^__\\^_\\^_\\^___\\^_\\^_\\^____\\^_\\^_\\^"
    "fish" "10101")

make_test(
    test_morse_encoding 
    "33 __\\^_\\^_\\^_\\^__\\^_\\^_\\^___\\^_\\^_\\^____\\^_\\^_\\^"
    "FISH" "10101")
    

# Test for working on unknown characters (replace unknown by 0 before encoding)
make_test(
    test_morse_encoding 
    "30 \\^\\^\\^\\^\\^\\^_\\^_\\^\\^\\^\\^\\^\\^\\^_\\^_\\^\\^\\^\\^\\^\\^\\^_\\^_\\^"
    "000" "10101")

make_test(
    test_morse_encoding 
    "30 \\^\\^\\^\\^\\^\\^_\\^_\\^\\^\\^\\^\\^\\^\\^_\\^_\\^\\^\\^\\^\\^\\^\\^_\\^_\\^"
    "$!?" "10101")

# Test for working on empty strings
make_test(
    test_morse_encoding 
    "0 "
    "E_M_P_T_Y" "10101")

### EQUALIZE LENGTH ###

# first signal is 15 time steps, second is 8
# both are generated with cyclic generator of mod 3 and offset 1
make_test(
    test_equalize_length 
"2 signals[\\r\\n\\t ]*
15 _\\^__\\^__\\^__\\^__\\^_[\\r\\n\\t ]*
8 _\\^__\\^__\\^[\\r\\n\\t ]*
2 signals[\\r\\n\\t ]*
15 _\\^__\\^__\\^__\\^__\\^_[\\r\\n\\t ]*
15 _\\^__\\^__\\^_______"    
    15 8 3 1
)

make_test(
    test_equalize_length
"2 signals[\\r\\n\\t ]*
5 _\\^__\\^[\\r\\n\\t ]*
5 _\\^__\\^[\\r\\n\\t ]*
2 signals[\\r\\n\\t ]*
5 _\\^__\\^[\\r\\n\\t ]*
5 _\\^__\\^"
    5 5 3 1 ) 

make_test(
    test_equalize_length 
"2 signals[\\r\\n\\t ]*
0[\\r\\n\\t ]* 
0[\\r\\n\\t ]* 
2 signals[\\r\\n\\t ]*
0[\\r\\n\\t ]* 
0"
    0 0 3 1)