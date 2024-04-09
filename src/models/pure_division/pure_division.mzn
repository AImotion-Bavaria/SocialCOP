enum Agents = {a, b, c};
int: n_items; 

set of int: Items = 1..n_items; 
array[Agents, Items] of int: valuation; 

% every item has to go to one agent
array[Items] of var Agents: assigned_to; 

% need a function for the share of every agent
include "int_set_channel.mzn";

function var set of Items: share_for(var Agents: a_) = 
let {
  array[Agents] of var set of Items: assigned_items;
  constraint int_set_channel(assigned_to, assigned_items);
} in 
  assigned_items[a_];
  
function var int: private_utility_for(var Agents: a_, var set of Items: items) = 
  sum(i in items) (valuation[a_, i])
;

% just for output purposes
array[Agents] of var set of Items: shares;
constraint shares =  [ share_for(a_) | a_ in Agents];
array[Agents] of var int: utils;
constraint utils =  [ private_utility_for(a_, shares[a_]) | a_ in Agents];

% output ["Assigned_to: \(assigned_to)\nShares = \(shares)\nUtils = \(utils)"];