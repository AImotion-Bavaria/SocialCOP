% Towards https://en.wikipedia.org/wiki/Envy-freeness:

enum Days = {Mon, Tue, Wed, Thu, Fri};
enum Tables = {NoTable, Table1, Table2};
set of int: Agents = {1,2};

array[Agents, Days] of var Tables: assigned;
% Overall constraint: no table can be assigned to 2 agents; exception: both can get NoTable
constraint forall (d in Days) ((assigned[1, d] != assigned[2, d]) \/ (assigned[1,d] == NoTable));

% ----------------------------------------------
% A util function to "slice out" the assignment of some agent of the overall assignment
% for better readability
function array[Days] of var Tables: get_assigned_for(Agents: a) = 
let {
  array[Days] of var Tables: assigned_for_a;
  constraint assigned_for_a = [assigned[a, d] | d in Days]
}
  in
  assigned_for_a
;

% ----------------------------------------------
% Generic preference-type definitions:
function var bool: wants_some_day(array[Days] of var Tables: assigned_for_agent,
                                  Days: day
                                  )
          =  (assigned_for_agent[day] != NoTable);   

function var bool: wants_some_number_days(
                         array[Days] of var Tables: assigned_for_agent,
                         int: num_days)
          =  (sum(d in Days) (bool2int(assigned_for_agent[d] != NoTable)) == num_days);

function var bool: wants_a_break(
                         array[Days] of var Tables: assigned_for_agent) =
                 (not (exists(d in Days diff {Fri}) ( 
                   (assigned_for_agent[d] != NoTable) /\ 
                   (assigned_for_agent[enum_next(Days, d)] != NoTable)
)));

% this type of constraint is interesting - but what would it mean for envy-freeness?
% How do I prefer my share of tables to that of another agent, esp. the agent 
% who I want to sit with? Of course, if they have a table on Thursday, I'd like that too
% I could only envy a third agent that has a shared day with my partner
% Conclusion: I'm ignoring it just for now 
function var bool: wants_shared_day(
                  array[Days] of var Tables: assigned_for_agent,
                  array[Days] of var Tables: assigned_for_partner)
          =   (exists(d in Days) ( 
                  (assigned_for_agent[d] != NoTable) /\
                  (assigned_for_partner[d] != NoTable)
));
% ----------------------------------------------
% AGENT 1
% Preferences 1 :

% ----------------------------------------------
% 1: will Dienstag haben, 
% 2: will zwei Tage haben, 
% 3: will einen Tag Pause dazwischen

function var int: utility_agent_1(array[Days] of var Tables: assigned_for_agent) =
let {
  int: nScs = 3;
  set of int: softConstraints = 1..nScs;
  array[softConstraints] of var bool: sat; 
  
  % 1: will Dienstag haben, 
  constraint sat[1] = wants_some_day(assigned_for_agent, Tue);
  % 2: will zwei Tage haben, 
  constraint sat[2] == wants_some_number_days(assigned_for_agent, 2);
  % 3: will einen Tag Pause dazwischen (keine zwei aufeinanderfolgenden Tage)     
  constraint sat[3] == wants_a_break(assigned_for_agent);

  % some weights for the soft constraints
  array[softConstraints] of int: weights = [3, 2, 1];
  var 0..sum(weights): utility = sum(s in softConstraints) (weights[s] * bool2int(sat[s]));
} in 
  utility
;                                       

% AGENT 2
% preferences 1 :
% 1: will Donnerstag haben, 
% 2: will drei Tage haben, 
% 3: will min. einen Tag gemeinsam mit Agent1
function var int: utility_agent_2(array[Days] of var Tables: assigned_for_agent) =
let {
  int: nScs = 3;
  set of int: softConstraints = 1..nScs;
  array[softConstraints] of var bool: sat; 
  
  % 1: will Dienstag haben, 
  constraint sat[1] = wants_some_day(assigned_for_agent, Thu);
  % 2: will drei Tage haben, 
  constraint sat[2] == wants_some_number_days(assigned_for_agent, 3);
  % 3: will min. einen Tag gemeinsam mit Agent1     
  constraint sat[3] == wants_a_break(assigned_for_agent);

  % some weights for the soft constraints
  array[softConstraints] of int: weights = [3, 2, 1];
  var 0..sum(weights): utility = sum(s in softConstraints) (weights[s] * bool2int(sat[s]));
} in 
  utility
;

% ----------------------------------------------
% ----------------------------------------------
% Utility overall

% first a "dirty" hack to emulate a function pointer: a big if-then-cascade
function var int: util_switch(int: function_id, 
                              array[Days] of var Tables: assigned_for_agent) =
if function_id == 1 then 
  utility_agent_1(assigned_for_agent)
else 
  utility_agent_2(assigned_for_agent)
endif;

array[Agents] of var int: utilities;
constraint utilities = [util_switch(a, get_assigned_for(a)) | a in Agents];

% Envy-freeness is defined as : There are no agents $i$ and $j$ such that
% $V_i(X_i) < V_i(X_j)$

predicate envy_free() =
  forall(i in Agents, j in Agents where i != j) (
       util_switch(i, get_assigned_for(i)) >= util_switch(i, get_assigned_for(j))
);

constraint envy_free();
  
solve maximize sum(utilities);