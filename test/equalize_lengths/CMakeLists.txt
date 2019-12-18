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