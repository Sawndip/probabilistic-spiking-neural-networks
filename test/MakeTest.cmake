# The main function for making a new test case.
# It takes two required arguments:
# $target - Which executable do we need to invoke for this test case
# $result - RegEx pattern to match or NO_REGEX if testing only successfull non-zero execution
# The other provided arguments are sent to the target executable.
# In addition it relies on a latent state via the hidden variable ${test_dir}
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
        add_executable(${target} ${test_dir}/${target}.cpp)

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