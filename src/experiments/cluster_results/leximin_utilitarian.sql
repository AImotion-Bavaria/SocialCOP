SELECT r1.configuration, r1.sum_utility as "utility_r1", 
      r1.utility_vector as "leximin_utility",
      r2.utility_vector as "utilitarian_utility",
      r2.configuration, r2.sum_utility as "utility_r2",
	r2.sum_utility/r1.sum_utility as "quotient" FROM 
Results r1 INNER JOIN Results r2
ON r1.model == r2.model AND 
   r1.data_files == r2.data_files AND
   r1.solver == r2.solver AND 
   r1.configuration != r2.configuration AND 
   r1.configuration == "leximin"