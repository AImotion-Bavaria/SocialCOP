% Use this editor as a MiniZinc scratch book

set of int: Site;
Site: hotel;
array[Site,Site] of int: distance;
int: num_sites;

array[Site] of int: durationOfVisit;
array[Site] of int: siteOpens;
array[Site] of int: siteCloses;

array[Site] of var Site: next;

include "subcircuit.mzn";
constraint subcircuit(next);

array[Site] of var 0..infinity: arrivalTime;
array[Site] of var 0..infinity: startOfVisit;

constraint arrivalTime[next[hotel]] = distance[hotel,next[hotel]];
constraint forall (i in Site where i != hotel /\ i != next[i]) (
  arrivalTime[next[i]] = startOfVisit[i]+durationOfVisit[i]+distance[i,next[i]]
);

constraint forall (i in Site where i != next[i]) (
   startOfVisit[i] = max(arrivalTime[i], siteOpens[i])
/\ startOfVisit[i]+durationOfVisit[i] <= siteCloses[i]
);

constraint forall (i in Site where i = next[i]) (
  startOfVisit[i]=0 /\ arrivalTime[i]=0
);

var int: travel_time = sum (i in Site where i != next[i]) (distance[i,next[i]]);

int: num_passengers;
set of int: Passenger = 1..num_passengers;
array[Passenger,Site] of int: site_score;
array[Passenger] of int: preferred_n_sites;
array[Passenger] of int: preferred_travel_time;
array[Passenger,1..num_sites] of int: preference_weights;

array[Passenger,1..num_sites] of var int: passenger_scores;
constraint passenger_scores  = 
array2d(Passenger,1..num_sites,
[ if i=1 then sum (s in Site where s != hotel /\ s != next[s]) (site_score[p,s])
  elseif i=2 then (preferred_n_sites[p]-abs(preferred_n_sites[p] - sum (s in Site where s != hotel) (s != next[s])))
  else min(preferred_travel_time[p] - travel_time,0)
  endif
| p in Passenger, i in 1..num_sites]);

array[Passenger] of var int: score;
constraint score  =
[  sum([ preference_weights[p,i] * passenger_scores[p,i] | i in 1..num_sites])
  | p in Passenger,  ];
  
  % for loop with num_sites?
