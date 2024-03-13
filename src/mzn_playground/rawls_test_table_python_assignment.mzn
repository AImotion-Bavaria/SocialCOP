include "table.mzn";

int: dimension;
int: possibilities = max(index_set_1of2(values));

%var int: index;
set of int: Agents = 1..dimension;
array[int,int] of int: values;

array[Agents] of var 0..100: utilities;
%assignment of values to utilities
constraint table(utilities,values);

var 0..100: worst_util;
var Agents: worst_agent;

%array[Agents] of var 0..100: minimal_values;
array[int] of int: maxmin_values; % initially [], then [30], then [30, 98]
array[index_set(maxmin_values)] of var Agents: maxmin_agents;

include "alldifferent.mzn";
constraint alldifferent(maxmin_agents);

constraint forall(i in index_set(maxmin_values))(utilities[maxmin_agents[i]] == maxmin_values[i] );

var set of Agents: maxmin_taken;
maxmin_taken = {maxmin_agents[i] | i in index_set(maxmin_values)};

constraint worst_util = min (a in Agents diff maxmin_taken) (utilities[a]);

constraint utilities[worst_agent] = worst_util;
constraint not(worst_agent in maxmin_taken);

solve maximize worst_util;
