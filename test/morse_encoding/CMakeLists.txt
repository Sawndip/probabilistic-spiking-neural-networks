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