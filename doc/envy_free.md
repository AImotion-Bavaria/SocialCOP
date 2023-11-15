# Envy-Freeness:
Envy freeness is given if the Value function $V_i$ for every agent i is resulting the highest result with their given ressources when compared to every other agents ressources. 

According to a valuation function:
$X \succeq_i Y \leftrightarrow V_i(X) \geq V_i(Y)$

Envy-freeness is defined as : There are no agents $i$ and $j$ such that
$V_i(X_i) < V_i(X_j)$

Therefore the best solution is calculated and afterwards the envy freeness is calculated

An envy-free solution is a fair solution 

Still keeping in mind that the overall value should be maximized ($sum(V_i)$ should be max)
If this is not given, if every agent gets 0, it would still be envy-free and considered fair. But of course it would not be a suitable solution.

If there is no envy-free solution besides every agent getting 0, the given constraints can not lead to a completly fair solution.

Transferred to Table situation:

Goal: Input form with placeholder for specific constraints, e.g. Days to get the table,
preferences of constraints
--> simulated in minizinc 

Afterwards envy-freeness is calculated and returns if a fair solution is possible and which solutions.

--> value to be optimized is combination of envy-freeness and utility

# Grant Proposal
https://research.google/outreach/research-scholar-program/