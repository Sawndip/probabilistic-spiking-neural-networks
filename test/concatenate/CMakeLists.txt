# The four parameters are number of time steps for the 4 different signals
make_test(
    test_concatenate
"4 signals
0 
0 
0 
5 \\^___\\^
1 signals
5 \\^___\\^"
    0 0 0 5) 

make_test(
    test_concatenate 
"4 signals
0 
0 
5 \\^___\\^
0 
1 signals
5 \\^___\\^"    
    0 0 5 0)

make_test(
    test_concatenate 
"4 signals
0 
5 \\^___\\^
0 
0 
1 signals
5 \\^___\\^" 
    0 5 0 0)

make_test(
    test_concatenate 
"4 signals
5 \\^___\\^
0 
0 
0 
1 signals
5 \\^___\\^" 
    5 0 0 0)

make_test(
    test_concatenate 
"4 signals
0 
0 
0 
0 
1 signals
0 " 
    0 0 0 0)

make_test(
    test_concatenate 
"4 signals
15 \\^___\\^___\\^___\\^__
10 \\^___\\^___\\^_
5 \\^___\\^
0 
1 signals
30 \\^___\\^___\\^___\\^__\\^___\\^___\\^_\\^___\\^"    
    15 10 5 0
)