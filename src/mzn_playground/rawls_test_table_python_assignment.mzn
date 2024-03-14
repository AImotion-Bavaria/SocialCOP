include "table.mzn";

int: dimension;
int: possibilities = max(index_set_1of2(values));

%var int: index;
set of int: Agents = 1..dimension;
array[int,int] of int: values;

array[Agents] of var 0..100: utilities;
%assignment of values to utilities
constraint table(utilities,values);

var 0..100: next_worst_util;
var Agents: next_worst_agent;

% maxmin_values are the maximized minimal values we received over time
% this _has to be_ an increasing array (otherwise a previously found value would not be the worst-off agent) 
array[int] of int: maxmin_values; % initially [], then [30], then [30, 98]
% for every maxmin_value, we need a corresponding agent (can only appear once -> alldifferent) 
array[index_set(maxmin_values)] of var Agents: maxmin_agents;
include "alldifferent.mzn";
constraint alldifferent(maxmin_agents);
constraint forall(i in index_set(maxmin_values))(utilities[maxmin_agents[i]] == maxmin_values[i] );

% figure out which agents already play a role in the chain of maxmin values
var set of Agents: agent_maxmin_taken;
agent_maxmin_taken = {maxmin_agents[i] | i in index_set(maxmin_values)};

constraint next_worst_util = min (a in Agents diff agent_maxmin_taken) (utilities[a]);

constraint utilities[next_worst_agent] = next_worst_util;
constraint not(next_worst_agent in agent_maxmin_taken);

solve maximize next_worst_util;
