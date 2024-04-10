% a mixin that adds all primitives we need for envy-freeness

predicate envy_free() =
  envy_pairs == 0;

array[${AGENTS_ARRAY}, ${AGENTS_ARRAY}] of var int: envy_matrix; 
constraint forall (i, j in ${AGENTS_ARRAY}) (
       envy_matrix[i,j] = if i == j then 0 else 
                            bool2int( ${SHARE_UTIL_AGENT}(i, ${SHARE_FUNCTION}(i)) < ${SHARE_UTIL_AGENT}(i, ${SHARE_FUNCTION}(j)) )
                          endif
  );
var int: envy_pairs; 
constraint envy_pairs =  
  sum(i in ${AGENTS_ARRAY}, j in ${AGENTS_ARRAY} where i != j) (
       envy_matrix[i, j]
  );

% can be enforced as: constraint envy_free();
