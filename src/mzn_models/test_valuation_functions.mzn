function var int: double_me(var int: x) =
  2 * x;

 function var int: triple_me(var int: x) =
  3 * x;
  
var 1..10: a;
var 1..10: b;

var 1..10: x;
constraint x = 4; % just a test 

% I want something like this 
%array[1..2] of function: util_funcs = [double_me, triple_me];
%array[1..2] of var int: utils = [util_funcs[i](x) | i in 1..2];

function var int: generic_switcher(int: function_id, var int: x) =
if function_id == 1 then 
  double_me(x)
else 
  triple_me(x)
 endif;

array[1..2] of var int: y;
constraint y = [generic_switcher(i, x) | i in 1..2];