include(MakeTest.cmake)

# Make a multitude of new tastes for a feature represented by a directory $dir_name.
function (add_test_dir dir_name)
    # A small hack around use of many directories so everything compiles
    set(test_dir ${dir_name})

    include(${dir_name}/CMakeLists.txt)
endfunction()

add_test_dir(cyclic_generator)

add_test_dir(morse_encoding)

add_test_dir(equalize_lengths)

add_test_dir(concatenate)