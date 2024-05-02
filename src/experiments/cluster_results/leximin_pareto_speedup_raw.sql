SELECT * FROM 
Results r1 INNER JOIN Results r2
ON r1.model == r2.model AND 
   r1.data_files == r2.data_files AND
   r1.solver == r2.solver AND 
   r1.configuration != r2.configuration AND 
   r1.configuration == "leximin"