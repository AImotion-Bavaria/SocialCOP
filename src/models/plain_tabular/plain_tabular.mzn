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
set of int: Columns = 1..n_agents+1; % the first entry is the solution index, the others for agents

% plain tabular also allows for specifiying the order in which
% solutions are draws
array[int, Columns] of par int: example_solution_utils;
array[Columns] of var int: drawn_solution;

% write the first column into solution_index to make search deterministic
var int: solution_index;
constraint solution_index = drawn_solution[1];

% take on of the rows of the given solutions = index + utils
constraint table(drawn_solution, example_solution_utils);

constraint forall(i in Agents) (drawn_solution[i+1] = utilities[i]);

ann: deterministic_search = int_search([solution_index], input_order, indomain_min, complete);

%solve :: deterministic_search satisfy;