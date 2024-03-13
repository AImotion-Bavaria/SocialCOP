% Parameters - they are fixed
% ==============================================
% There are n agents
% m out of which are selected to get a bonus of 1
int: n;
int: m;
set of int: Agents = 1..n;

% Variables: What I'm interested in
% ==============================================
array[Agents] of var 0..1: selected;

% Constraints: What are the conditions
% ==============================================
% exactly m need to be 1
constraint sum(a in Agents) (selected[a]) == m;

solve
:: int_search(selected, input_order, indomain_random, complete)
satisfy;