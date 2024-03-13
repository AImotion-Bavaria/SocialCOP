include "table.mzn";

set of int: Agents = 1..4;

array[Agents] of var 0..100: utilities;

constraint table(utilities, 
                 [| 50, 20, 30, 99 |
                    39, 18, 12, 99 |
                    42, 37, 80, 12 |
                    20, 80, 80, 80|]);
                    
% 1. Rawls -> maximin (results in 20)
%solve maximize worst_util;

var int: worst_util;
var Agents: worst_agent;

constraint worst_util = min(utilities);
constraint utilities[worst_agent] = worst_util;

% after 1. Rawls :
constraint worst_util >= 20;

solve maximize min([utilities[a] | a in Agents where a != worst_agent] );
