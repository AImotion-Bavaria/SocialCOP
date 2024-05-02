SELECT r1.solver, sum(r1.envy_pairs), sum(r2.envy_pairs), Sum(r1.sum_utility), Count(*) FROM 
Results r1 INNER JOIN Results r2
ON r1.model == r2.model AND 
   r1.data_files == r2.data_files AND
   r1.solver == r2.solver AND 
   r1.configuration != r2.configuration AND 
   r1.configuration == "utilitarian"
 Group By r1.solver