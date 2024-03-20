% Define input parameters
array[int, int] of int: utilities;

int: dimension;
int: m;

array[1..dimension] of var float: max_values; % Use float to allow division
array[1..dimension] of var int: selected;

constraint forall (p in 1..dimension) (
    max_values[p] = max([utilities[i, p] | i in 1..m]) / dimension
);

constraint forall (p in 1..dimension) (
    selected[p] >= max_values[p]
);


constraint exists(i in 1..m) (
    forall(j in 1..dimension) (
        selected[j] = utilities[i, j]
    ));
    
%maybe utilitarian approach should be added to get maximum possible overall value 
    

% Solve statement
solve satisfy;

