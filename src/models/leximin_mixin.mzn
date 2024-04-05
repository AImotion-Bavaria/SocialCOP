set of int: Agents_Rawls_Mixin;
array[Agents_Rawls_Mixin] of var int: utilities_Rawls_Mixin; 

var int: next_worst_util;
var Agents_Rawls_Mixin: next_worst_agent;

% maxmin_values are the maximized minimal values we received over time
% this _has to be_ an increasing array (otherwise a previously found value would not be the worst-off agent) 
array[int] of int: maxmin_values; % initially [], then [30], then [30, 98]
% for every maxmin_value, we need a corresponding agent (can only appear once -> alldifferent) 
array[index_set(maxmin_values)] of var Agents_Rawls_Mixin: maxmin_agents;
include "alldifferent.mzn";
constraint alldifferent(maxmin_agents);
constraint forall(i in index_set(maxmin_values))(utilities_Rawls_Mixin[maxmin_agents[i]] == maxmin_values[i] );

% figure out which agents already play a role in the chain of maxmin values
var set of Agents_Rawls_Mixin: agent_maxmin_taken;
agent_maxmin_taken = {maxmin_agents[i] | i in index_set(maxmin_values)};

constraint next_worst_util = min (a in Agents_Rawls_Mixin diff agent_maxmin_taken) (utilities_Rawls_Mixin[a]);

constraint utilities_Rawls_Mixin[next_worst_agent] = next_worst_util;
constraint not(next_worst_agent in agent_maxmin_taken);

solve maximize next_worst_util;