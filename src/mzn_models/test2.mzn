set of int: Agents = 1..2;
set of int: Items = 1..4;

array[Agents, Items] of var bool: assigned_to;
array[Agents, Items] of int: likes = [| 2, 5, 4, 8 | 9, 3, 3, 1 |];

constraint forall(i in Items) (sum(a in Agents)(assigned_to[a,i]) <= 1); 


% Agent1:


% Agent2: 
% sc1: x < 4
% sc2: x + y > 7