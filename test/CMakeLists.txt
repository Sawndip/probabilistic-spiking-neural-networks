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

    set_tests_properties(${test_name}
        PROPERTIES PASS_REGULAR_EXPRESSION ${result}
    )
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