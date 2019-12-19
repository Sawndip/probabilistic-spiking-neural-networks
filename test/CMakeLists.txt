include(MakeTest.cmake)

# Make a multitude of new tastes for a feature represented by a directory $dir_name.
function (add_test_dir dir_name)
    # A small hack around use of many directories so everything compiles
    set(test_dir ${dir_name})

    include(${dir_name}/CMakeLists.txt)
endfunction()

set (test_dirs 
    # list of directories for whom tests should be conducted
    cyclic_generator 
    morse_encoding
    equalize_lengths
    concatenate
    merge)

foreach(td ${test_dirs})
    add_test_dir(${td})
endforeach(td in ${test_dirs})