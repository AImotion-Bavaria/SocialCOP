SELECT r1.solver, 1.0*sum(r1.envy_pairs > 0) / Count(*) as "UT Instances with Envy Pairs", sum(r2.envy_pairs) as "UT+EF Instances with Envy Pairs", 
                  avg(r1.solving_runtime) as "UT Runtime", avg(r2.solving_runtime) as "UT+EF Runtime" FROM 
Results r1 INNER JOIN Results r2
ON r1.model == r2.model AND 
   r1.data_files == r2.data_files AND
   r1.solver == r2.solver AND 
   r1.configuration != r2.configuration AND 
   r1.configuration == "utilitarian"
 Group By r1.solver