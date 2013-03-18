PP_miniproject: ACO
==============

CPU implementation:

ACO:

1-Go to the ACO CPU directory (cd CPU/ACO)
2-There is two versions:
	-A version that create a very simple graph where each node is connected(ACO.c):
	1-to compile: gcc ACO.c -lm -o ACO
	2-to run:  ACO
	-A version that load the graph and the pheromone via a file located in the data directory(ACO_load.c):
	1-to compile: gcc ACO_load.c -lm -o ACO_load
	2-to run:     ACO_load

	3-In this version there two examlples tested. If you want to change go to line 27 and 31 and change the name of the file.

3-For both version you can change the set of parameters as you wish in the file constant.h.

Cube:






GPU implementation:

1-Go to the GPU directory (cd GPU)
2-Edit the Makefile and change the source file that you want to compile at line 11
3-To compile: make
4-To run :    bin/release/smb

On the GPU implementation we just applied the ACO to a graph where all nodes are connected.

Note:
-In all version we can change the parameters between line 20 and 32 as you wish.

For the other parameters we have to make sure that these two point are respected:
-The three first version are not multiple block. Hence GRID_SIZE is always 1 and BLOCK SIZE has to be the same value as GRAPH_SIZE.

-In the last version which is multiple block you can change the size of the block and the grid.
But make sure that BLOCK_SIZE*GRID_SIZE=GRAPH_SIZE. 
