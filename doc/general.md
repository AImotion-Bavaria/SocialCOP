# General Notes:

| Problem Class   |Single-Shot                           | Repeated              |
|-----------------|--------------------------------------|-----------------------|
|Division Problem |Envy-freeness, Proportionality        |Gini, Artificial Karma, Fairness Index |
|Shared Decision  |Copeland, Approval Voting, Condorcrete, Majority|Gini, Artificial Karma, Fairness Index |


## Agent A wants to be in the office the same day as agent B:
This can not be handled with envy-freeness (no possibility of comparing agent A's distribution to agent B's), gini (the equal distribution does not give any assumptions about the shared days) or proportionality (if Agent A was the only Agent, he could not share the office day with B).

It can be viewed as shared decision problem:
The office is the shared variable. The days Agent A gets are equal to the sights visited by multiple buses. It can be treated similarly to the scenario of a group of visitors being splitted into different buses to visit sights.