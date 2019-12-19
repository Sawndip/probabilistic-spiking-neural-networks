# The four parameters are number of time steps for the 4 different signals
make_test(
    test_merge
"2 signals
0 
0 
2 signals
0 
5 \\^___\\^
4 signals
0 
0 
0 
5 \\^___\\^"
    0 0 0 5) 

make_test(
    test_merge 
"15 \\^___\\^___\\^___\\^__
10 \\^___\\^___\\^_
2 signals
5 \\^___\\^
0 
4 signals
15 \\^___\\^___\\^___\\^__
10 \\^___\\^___\\^_
5 \\^___\\^
0 "    
    15 10 5 0
)