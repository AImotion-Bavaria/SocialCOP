SELECT  DISTINCT r1.model, r1.data_files,
	  r1.sum_utility as "sum_utility_leximin", 
     r1.max_utility as "max_leximin",
	 r1.min_utility as "min_leximin",
	 r1.utility_vector as "leximin_utility_vector",
      r2.sum_utility as "sum_utility_utilitarian", 
     r2.max_utility as "max_utilitarian",
	 r2.min_utility as "min_utilitarian",
	 r2.utility_vector as "utilitarian_utility_vector",
	 r3.sum_utility as "sum_utility_rawls", 
     r3.max_utility as "max_rawls",
	 r3.min_utility as "min_rawls",
	 r3.utility_vector as "rawls_utility_vector"
	FROM 
Results r1 INNER JOIN Results r2 INNER JOIN Results r3
ON r1.model == r2.model AND r2.model == r3.model AND 
   r1.data_files == r2.data_files AND r2.data_files == r3.data_files AND
   r1.solver == r2.solver AND r2.solver == r3.solver AND 
   r1.configuration != r2.configuration AND r2.configuration != r3.configuration AND r1.configuration != r3.configuration AND 
   r1.configuration == "leximin" AND
   r2.configuration =="utilitarian"
   WHERE r1.solver="chuffed"