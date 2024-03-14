% Parameters - they are fixed
% ==============================================
% There are n agents
% m out of which are selected to get a bonus of 1
int: n;
int: m;
set of int: Agents = 1..n;
array[Agents] of int: weights; 
% default is [1, 1, ..., 1]

% Variables: What I'm interested in
% ==============================================
array[Agents] of var 0..1: selected;
var 0 .. n*max(weights): selection_weight; 
array[Agents] of var 0..max(weights): utilities = [selected[a] * weights[a] | a in Agents];

% Constraints: What are the conditions
% ==============================================
% exactly m need to be 1
constraint sum(a in Agents) (selected[a]) == m;
constraint selection_weight = sum(a in Agents)
                          (selected[a] * weights[a]);




                          