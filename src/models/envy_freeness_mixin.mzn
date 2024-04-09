% a mixin that adds all primitives we need for envy-freeness

predicate envy_free() =
  envy_pairs() == 0;

function var int: envy_pairs() = 
  sum(i in ${AGENTS_ARRAY}, j in ${AGENTS_ARRAY} where i != j) (
       bool2int( ${SHARE_UTIL_AGENT}(i, ${SHARE_FUNCTION}(i)) >= ${SHARE_UTIL_AGENT}(i, ${SHARE_FUNCTION}(j)) )
  );

% can be enforced as: constraint envy_free();