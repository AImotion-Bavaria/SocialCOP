include "table.mzn";

int: dimension;
int: possibilities;
var int: index;
set of int: Agents = 1..dimension;
array[1..possibilities,Agents] of int: values;
array[Agents] of var 0..100: utilities;
%assignment of values to utilities
constraint table(utilities,values);

var 0..100: worst_util;
var Agents: worst_agent;

array[Agents] of var 0..100: minimal_values;
%constraint table(minimal_values, 
 %                [|99, 0, 0, 0|]);

%only evaluate preselected solutions; valid solutions contain every min_value at least one time, if there is the same value multiple times it has to be differentiated
%starts with index 2 as first iteration minimal_values is equal to [0,0,0,0], 0 is not necessarily included
constraint if index>1 then 
  forall(i in 2..index) (
  let {
    var int: count_mv_u = sum(a in utilities) (a = minimal_values[i-1]);
    var int: count_equal_values = count_eq(minimal_values,minimal_values[i-1]);
  } in
  %if not all values of same amount are stored in minimal_values yet the count can be higher than count_equals
  if minimal_values[i-1]=max(minimal_values) then 
    count_mv_u >= count_equal_values
    %if value is not highest number stored in minimal_values it has to be the exact same amount
    else count_mv_u=count_equal_values
    endif
) endif;


var int: num_max_values = count_eq(utilities,max(minimal_values));
%constraint worst_util = min({x | x in utilities where ((x > max(minimal_values)))});
constraint if num_max_values<2 then
 worst_util = min({x | x in utilities where ((x > max(minimal_values)))})
 else  worst_util = min({x | x in utilities where (x = max(minimal_values))})
 endif;
constraint utilities[worst_agent] = worst_util;

solve maximize worst_util;
