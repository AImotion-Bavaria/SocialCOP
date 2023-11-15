% Towards https://en.wikipedia.org/wiki/Envy-freeness:

enum Days = {Mon, Tue, Wed, Thu, Fri};
enum Tables = {NoTable, Table1, Table2};
set of int: Agents = {1,2};

array[Agents, Days] of var Tables: assigned;
int: nSoftConstraints = 3;
set of int: SoftConstraints = 1..nSoftConstraints;
array[Agents, SoftConstraints] of var bool: satisfied;

%overall constraint: no table can be assigned to 2 agents; exception: both can get NoTable
constraint forall (d in Days) ((assigned[1, d] != assigned[2, d]) \/ (assigned[1,d] == NoTable));

% preferences 1 :
% 1: will Dienstag haben, 
% 2: will zwei Tage haben, 
% 3: will einen Tag Pause dazwischen


% 1: will Dienstag haben
function var bool: wants_some_day(Agents: agent_id, 
                         Days: day)
          =  (assigned[agent_id, day] != NoTable);

constraint satisfied[1, 1] == wants_some_day(1, Tue);

% 2: will zwei Tage haben, 
function var bool: wants_some_number_days(
                         Agents: agent_id, 
                         int: num_days)
          =  (sum(d in Days) (bool2int(assigned[agent_id, d] != NoTable)) == num_days);

constraint satisfied[1, 2] == wants_some_number_days(1, 2);

% 3: will einen Tag Pause dazwischen (keine zwei aufeinanderfolgenden Tage)     
function var bool: wants_a_break(
                         Agents: agent_id) =
                 (not (exists(d in Days diff {Fri}) ( 
                   (assigned[agent_id, d] != NoTable) /\ (assigned[agent_id, enum_next(Days, d)] != NoTable)
)));
                                                  
constraint satisfied[1, 3] == wants_a_break(1);

%AGENT2
% preferences 1 :
% 1: will Donnerstag haben, 
% 2: will drei Tage haben, 
% 3: will min. einen Tag gemeinsam mit Agent1
constraint satisfied[2, 1] == wants_some_day(2, Thu);
constraint satisfied[2, 2] == wants_some_number_days(2, 3);

% 3: will min. einen Tag gemeinsam mit Agent1

function var bool: wants_shared_day(
                         Agents: agent_id, 
                         Agents: partner_id)
          =   (exists(d in Days) ( 
                  (assigned[agent_id, d] != NoTable) /\
                  (assigned[partner_id, d] != NoTable)
));

constraint satisfied[2, 3] == wants_shared_day(2, 1);

% Envy-freeness is defined as : There are no agents $i$ and $j$ such that
% $V_i(X_i) < V_i(X_j)$
function var int: valuation(Agents: agent_id, Agents: other_id) =
  
;

solve maximize sum(satisfied);