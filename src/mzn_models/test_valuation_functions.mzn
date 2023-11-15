function var int: double_me(var int: x) =
  2 * x;

 function var int: triple_me(var int: x) =
  3 * x;
  
predicate sum_is_even(array[int] of var int: vals) =
    sum(vals) mod 2 == 0;
   
var 1..10: a;
var 1..10: b;

constraint b = double_me(a); 

constraint sum_is_even([a, b]);

