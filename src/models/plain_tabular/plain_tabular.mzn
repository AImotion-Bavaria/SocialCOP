% This model serves as a blueprint
% It just consists of a fixed number 
% of solutions (here, 4) with utilities for the 
% respective agents
% ------------------------------------------------
% We shall use it to test various solution runners 
include "table.mzn";

set of int: Agents = 1..4;
array[Agents] of var 0..100: utilities;

constraint table(utilities, 
                 [| 50, 20, 30, 99 |
                    39, 18, 12, 99 |
                    42, 37, 80, 12 |
                    20, 80, 80, 80|]);