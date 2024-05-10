SELECT r1.solver, r1.model, 
	  r1.configuration, r1.sum_utility as "utility_r1", 
      r1.utility_vector as "leximin_utility", r1.solving_runtime, 
      r2.configuration, r2.sum_utility as "utility_r2", 
	  r2.utility_vector as "utilitarian_utility", r2.solving_runtime,
      r3.configuration, r3.sum_utility as "utility_r3", 
	  r3.utility_vector as "rawls_utility", r3.solving_runtime,
	 
	r2.sum_utility/r1.sum_utility as "quotient" FROM 
Results r1 INNER JOIN Results r2 INNER JOIN Results r3
ON r1.model == r2.model AND r2.model == r3.model AND 
   r1.data_files == r2.data_files AND r2.data_files == r3.data_files AND
   r1.solver == r2.solver AND r2.solver == r3.solver AND 
   r1.configuration != r2.configuration AND r2.configuration != r3.configuration AND r1.configuration != r3.configuration AND 
   r1.configuration == "leximin" AND
   r2.configuration =="utilitarian"