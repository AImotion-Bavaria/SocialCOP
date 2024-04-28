array[int, ${AGENTS_ARRAY}] of int: previous_utilities; 
set of int: prev_util_index_set = index_set_1of2(previous_utilities);
  
constraint not exists(j in prev_util_index_set) ( % there must not be any previous solution whose utilities dominate the current solution's utils
              forall(i in ${AGENTS_ARRAY}) (
                                     ${UTILITY_ARRAY}[i] <= previous_utilities[j, i]
                                  )
              /\ 
              exists(i in ${AGENTS_ARRAY}) (${UTILITY_ARRAY}[i] < previous_utilities[j, i]) 
);