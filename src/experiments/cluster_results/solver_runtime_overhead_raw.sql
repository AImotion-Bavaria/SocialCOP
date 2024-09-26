SELECT  DISTINCT r1.model, r1.data_files,
	 r1.solving_runtime as "leximin_runtime",
	 r2.solving_runtime as "util_runtime",
	 r3.solving_runtime as "rawls_runtime",
	 r1.solving_runtime / r2.solving_runtime as "leximin_overhead",
	 r1.solver
	FROM 
Results r1 INNER JOIN Results r2 INNER JOIN Results r3
ON r1.model == r2.model AND r2.model == r3.model AND 
   r1.data_files == r2.data_files AND r2.data_files == r3.data_files AND
   r1.solver == r2.solver AND r2.solver == r3.solver AND 
   r1.configuration != r2.configuration AND r2.configuration != r3.configuration AND r1.configuration != r3.configuration AND 
   r1.configuration == "leximin" AND
   r2.configuration =="utilitarian"