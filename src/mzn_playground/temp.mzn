
%calculate satisfied array for ressources of agent2
array[1..3] of var bool: satisfied_envy_agent1 = array1d([(assigned_agent2[Tue] != NoTable), (sum(d in Days) (
                               bool2int(assigned_agent2[d] != NoTable)) == 2), (not (exists(d in Days diff {Fri}) ( 
                  (assigned_agent2[d] != NoTable) /\
                  (assigned_agent2[enum_next(Days, d)] != NoTable))))]);

%AGENT2
% preferences 1 :
% 1: will Donnerstag haben, 
% 2: will drei Tage haben, 
% 3: will min. einen Tag gemeinsam mit Agent1

array[Days] of var Tables: assigned_agent2;
int: nSoftConstraints_agent2 = 3;
set of int: SoftConstraints_agent2 = 1..nSoftConstraints_agent2;
array[SoftConstraints_agent2] of var bool: satisfied_agent2;

% 1: will Donnerstag haben
constraint satisfied_agent2[1] == (assigned_agent2[Thu] != NoTable);

% 2: will drei Tage haben, 
constraint satisfied_agent2[2] == (sum(d in Days) (
                               bool2int(assigned_agent2[d] != NoTable)) == 3);

% 3: will min. einen Tag gemeinsam mit Agent1                              
constraint satisfied_agent2[3] == (exists(d in Days) ( 
                  (assigned_agent2[d] != NoTable) /\
                  (assigned_agent1[d] != NoTable)
));

%calculate satisfied array for ressources of agent1

array[1..3] of var bool: satisfied_envy_agent2 = array1d([(assigned_agent1[Thu] != NoTable), (sum(d in Days) (
                               bool2int(assigned_agent1[d] != NoTable)) == 3),(exists(d in Days) ( 
                  (assigned_agent1[d] != NoTable) /\
                  (assigned_agent1[d] != NoTable)
))]);

%CALCULATE UTILITY

var int: overall_utility;
array[SoftConstraints_agent2] of int: utils_agent2 = [3, 2, 1];
array[SoftConstraints_agent1] of int: utils_agent1 = [3, 2, 1];

                                 
           
                                                                                      
                                                                                                                                                                                                                                               

                                
var bool: envy_free_agent1 = 
                      sum(s in SoftConstraints_agent1)(utils_agent1[s] * bool2int(satisfied_agent1[s]))
                       >=   sum(s in SoftConstraints_agent1)(utils_agent1[s] * bool2int(satisfied_envy_agent1[s]));
                      
var bool: envy_free_agent2 = sum(s in SoftConstraints_agent2)(utils_agent2[s] * bool2int(satisfied_agent2[s]))>=
                              sum(s in SoftConstraints_agent2)(utils_agent2[s] * bool2int(satisfied_envy_agent2[s]));
                              
var bool: envy_free = envy_free_agent1 /\ envy_free_agent2;

constraint overall_utility = bool2int(envy_free_agent1) * (sum(s in SoftConstraints_agent1) (
                                 utils_agent1[s] * bool2int(satisfied_agent1[s]))) + bool2int(envy_free_agent2) *  
                                 sum(sag2 in SoftConstraints_agent2) 
                                 (utils_agent2[sag2] * bool2int(satisfied_agent2[sag2]));



                      
output [ "Envy-Free: \(envy_free)\n",
         "Utility: \(overall_utility)\n",
         "Assigned Agent 1: \(assigned_agent1)\n",
         "Satisfied Agent 1: \(satisfied_agent1)\n", 
         "Assigned Agent 2: \(assigned_agent2)\n", 
         "Satisfied Agent 2: \(satisfied_agent2)\n", 
         "Envy Freeness Agent 1: \(envy_free_agent1)\n", 
         "Envy Freeness Agent 2: \(envy_free_agent2)\n" ];


solve maximize (overall_utility); 