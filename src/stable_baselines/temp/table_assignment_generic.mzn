% Table with opt
enum Days = {Mon, Tue, Wed, Thu, Fri};
enum Tables;
int: n_agents; 
set of int: Agents = 1..n_agents;

% Assignment of a table can be optional - some agents may not get a table at all
array[Agents, Days] of var opt Tables: assigned;

% Overall constraint: no table can be assigned to the same agent; exception: both can get no table
include "alldifferent.mzn";
constraint forall(d in Days) (alldifferent([assigned[a, d] | a in Agents]));

% ----------------------------------------------
% A util function to "slice out" the assignment of some agent of the overall assignment
% for better readability
function array[Days] of var opt Tables: get_assigned_for(Agents: a) :: promise_total = 
let {
  array[Days] of var opt Tables: assigned_for_a;
  constraint assigned_for_a = [assigned[a, d] | d in Days]
}
  in
  assigned_for_a
;

array  [Agents] of int: weights;
% Objective: Maximize the sum of weights of selected agents
var int: total_weight = sum(a in Agents) (sum(d in Days) (assigned[a,d]) * weights[a]);

% ----------------------------------------------
% Generic preference-type definitions:
function var bool: gets_some_day(array[Days] of var opt Tables: assigned_for_agent,
                                  Days: day
                                  )
          =  (occurs(assigned_for_agent[day]) );   

function var int: gets_that_many_days(
                         array[Days] of var opt Tables: assigned_for_agent
                         )
          =  sum(d in Days) (bool2int( occurs(assigned_for_agent[d]) ));

function var bool: wants_a_break_day(
                         array[Days] of var opt Tables: assigned_for_agent) =
                 (not (exists(d in Days diff {Fri}) ( 
                   (occurs(assigned_for_agent[d])) /\ 
                   (occurs(assigned_for_agent[enum_next(Days, d)]))
)));

% this type of constraint is interesting - but what would it mean for envy-freeness?
% How do I prefer my share of tables to that of another agent, esp. the agent 
% who I want to sit with? Of course, if they have a table on Thursday, I'd like that too
% I could only envy a third agent that has a shared day with my partner
% Conclusion: I'm ignoring it just for now UPDATE: In envy-freeness, it should not matter
% for overall utilities it's fine
function var bool: wants_shared_day(  
                  var Agents: agent, var Agents: partner)
          =   (exists(d in Days) ( 
                  (occurs(assigned[agent,d])) /\ (occurs(assigned[partner, d]))
));

% ----------------------------------------------
% Data for each agent to pick a set of these 
% generic preferences
% First, the share utils that are only dependent on the assigned share
array[Agents] of set of Days: wanted_days;
array[Agents] of set of int: wanted_number_days;
array[Agents] of bool: wanted_breaks;

% Second, utils that depend on the whole solution, i.e., also the shares of others
array[Agents] of set of Agents: desired_partners;

% ----------------------------------------------
% Now the actual utility functions
% functions are marked as promise_total since they need to be used in a reified context 
% (where an expression could be true or false) 
function var int: share_utility(Agents: a, array[Days] of var opt Tables: assigned_for_agent) :: promise_total =
let { 
  % 1: the wanted days 
  var int: wanted_days_satisfaction;
  % Can be at most 5, will be reduced for every day that I wanted but did not get
  constraint wanted_days_satisfaction = card(Days) - sum( w in wanted_days[a]) ( not( gets_some_day(assigned_for_agent, w) ));
  
  % 2: the wanted number of days , 
  var int: wanted_number_days_satisfaction;
  % wanted_number_days can be {} or a singleton set {i}, therefore we're summing over it
  % is at most 5, but could be reduced for any deviation between wanted and assigned days
  constraint wanted_number_days_satisfaction = card(Days) - sum( w in wanted_number_days[a]) ( abs(w - gets_that_many_days(assigned_for_agent)) );
  
  % 3: wanted breaks satisfaction
  var int: wanted_breaks_satisfaction;
  constraint wanted_breaks_satisfaction = if(wanted_breaks[a]) then wants_a_break_day(assigned_for_agent) else 1 endif;

} in 
  wanted_days_satisfaction + wanted_number_days_satisfaction + wanted_breaks_satisfaction
;                      

% these utilities come _on top_ of the already specified share utilities - may be none 
function var int: global_utility(Agents: a) =
let {
  var bool: works_with_desired_partner;
  int: max_possible_satisfaction = max(Agents);
  array[Agents] of var bool: has_shared_day_if_wanted; 
  
  constraint forall(b in Agents) (
    has_shared_day_if_wanted[b] = 
      if b in desired_partners[a] then 
        wants_shared_day(a, b)
      else 
        true 
      endif
  );
}
in 
  sum(has_shared_day_if_wanted)
;

% hard work from home constraints
int: max_wfh_days; % = 2;
include "count_leq.mzn";
constraint forall(a in Agents) (
  sum(d in Days) (absent(assigned[a, d])) <= max_wfh_days
);

% added arbitrary bounds due to compilation error in MiniZinc
array[Agents] of var -1000 .. 1000: utilities;
constraint utilities = [share_utility(a, get_assigned_for(a)) + global_utility(a) | a in Agents];

solve maximize sum(utilities)+total_weight;