% This model serves as a blueprint
% It just consists of a fixed number 
% of solutions (here, 4) with utilities for the 
% respective agents
% ------------------------------------------------
% We shall use it to test various solution runners 
include "table.mzn";

int: n_agents;
set of int: Agents = 1..n_agents;

array[Agents] of var 0..100: utilities;
array[int, Agents] of int: possible_solutions;

constraint table(utilities, 
                possible_solutions);
                    
