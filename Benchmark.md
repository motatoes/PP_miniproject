PP_miniproject
==============

Benchmark:

for this specific set of ACO parameters:

Number of iteration
ACO_ITER_MAX 10
//evaporation rate
EVAP_RATE 0.3
//influence rate of the pheromone
ALPHA 0.8
//influence rate of the heuristic (distance)
BETA 0.2
//Initial level of pheromone
INIT_PHEROMONE 5
//Update pheromone constant
UPDT_PHEROMONE_CONST 2

===========================

CPU implementation of ACO:
Nb_ant    : 1024
Graph_size: 1024
Time      : 3.766 s

=========================================

GPU extremely naive implementation of ACO:
Grid size : 1
Block size: 1024
Nb_ant    : 1024
Graph_size: 1024
Throughput: 0.0083 GFlop/s
Time      : 0.5027 s

==========================================

GPU 1st optimisation: (split everything into multiple kernel)
Grid size : 1
Block size: 1024
Nb_ant    : 1024
Graph_size: 1024
Throughput: 0.0309 GFlop/s
Time      : 0.2034 s

==========================================

GPU 2nd optimisation: (parallalize the function that calculate the best solution so far)
Grid size : 1
Block size: 1024
Nb_ant    : 1024
Graph_size: 1024
Throughput: 0.0320 GFlop/s
Time      : 0.1968 s

==========================================

GPU 3rd optimisation: (used multiple block for some of the kernels)
Grid size : 32
Block size: 32
Nb_ant    : 1024
Graph_size: 1024
Throughput: 0.0326 GFlop/s
Time      : 0.1932 s