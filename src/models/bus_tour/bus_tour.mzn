% Use this editor as a MiniZinc scratch book

set of int: Site;
Site: hotel;
array[Site,Site] of int: distance;
var int: maxTourDuration;
constraint maxTourDuration = sum(s1, s2 in Site where s1 < s2) (max(distance[s1, s2], distance[s2, s1]) );  

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

var int: travel_time;
constraint travel_time = sum (i in Site where i != next[i]) (distance[i,next[i]]);

int: num_passengers;
set of int: Passenger = 1..num_passengers;
int: num_category =3;
array[Passenger,Site] of int: site_score;
array[Passenger] of int: preferred_n_sites;
array[Passenger] of int: preferred_travel_time; % preferred maximum travel time
array[Passenger,1..num_category] of int: preference_weights;

array[Passenger,1..num_category] of var int: passenger_scores;
var int: number_of_visited_sites;
constraint number_of_visited_sites = sum (s in Site where s != hotel) (s != next[s]);

% number_of_visited_sites / preferred_n_sites[p] 
int: num_sites = card(Site);
constraint passenger_scores  = 
array2d(Passenger,1..num_category,
[ if i=1 then 
      sum (s in Site where s != hotel /\ s != next[s]) (site_score[p,s])
  elseif i=2 then 
      (num_sites - abs(preferred_n_sites[p] - number_of_visited_sites))
  else  
      % if travel_time (90) > preferred_max_travel_time (60) -> reduce some points
      % if preferred_max_travel_time (90) > travel_time (60) -> we are fully happy
      maxTourDuration - max(travel_time - preferred_travel_time[p], 0)
  endif
| p in Passenger, i in 1..num_category]);

array[Passenger] of var -10000..10000: score;
constraint score  =
[  sum([ preference_weights[p,i] * passenger_scores[p,i] | i in 1..num_category])
  | p in Passenger,  ];
  

% solve maximize sum(passenger_scores); 
