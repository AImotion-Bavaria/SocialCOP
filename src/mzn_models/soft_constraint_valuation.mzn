% Towards https://en.wikipedia.org/wiki/Envy-freeness:

enum Days = {Mon, Tue, Wed, Thu, Fri};
enum Tables = {NoTable, Table1, Table2};
% preferences 1 :
% 1: will Dienstag haben, 
% 2: will zwei Tage haben, 
% 3: will einen Tag Pause dazwischen

array[Days] of var Tables: assigned;
int: nSoftConstraints = 3;
set of int: SoftConstraints = 1..nSoftConstraints;
array[SoftConstraints] of var bool: satisfied;

% 1: will Dienstag haben
constraint satisfied[1] == (assigned[Tue] != NoTable);

% 2: will zwei Tage haben, 
constraint satisfied[2] == (sum(d in Days) (
                               bool2int(assigned[d] != NoTable)) == 2);

% 3: will einen Tag Pause dazwischen (keine zwei aufeinanderfolgenden Tage)                              
constraint satisfied[3] == (not (exists(d in Days diff {Fri}) ( 
                  (assigned[d] != NoTable) /\
                  (assigned[enum_next(Days, d)] != NoTable)
)));

var int: overall_utility;
array[SoftConstraints] of int: utils = [3, 2, 1];
constraint overall_utility = sum(s in SoftConstraints) (
                                 utils[s] * bool2int(satisfied[s])
);

solve maximize overall_utility; 