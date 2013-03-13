#include <cutil_inline.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


//cube libraries

//rubik's cube libs
#include "../CPU/cube/parse.c"
#include "../CPU/cube/move.c"

/*ACO parameters*/
	//Number of nodes in the graph
	#define GRAPH_SIZE 1024
	//Number of iteration in ACO algorithm
	#define ACO_ITER_MAX 10
	//evaporation rate
	#define EVAP_RATE 0.3
	//influence rate of the pheroneme
	#define ALPHA 0.8
	//influence rate of the heuristic (distance)
	#define BETA 0.2
	//Initial level of pheroneme
	#define INIT_PHERONEME 5
	//Update pheroneme constant
	#define UPDT_PHERONEME_CONST 2
	//Number of moves allowed through the graph
	#define NSTEPS 2
  //Number of ants
  #define NB_ANT 10
/*End ACO parameters*/

/*GPU parameters*/
	#define BLOCK_SIZE 64
	#define GRID_SIZE 1
	#define ITER_BENCHMARK 100
/*End GPU parameters*/


//function prototypes
float* sum_probability(int* h_graph, float* h_pheroneme, int size);
void update_probability(int* h_graph,float* h_pheroneme,float* h_probability, int size, float* sum)
;
//a macro function that takes as parameters the indexes
//of a 2d matrix and it's row size, and returns the 
//serialized index
#define SERIALIZE(i,j,row_size) i * row_size + j;


//functions prototypes
void datainit_graph(int*, int);
void datainit_pheroneme(float*, int);


__global__ void ACO_kernel(int* d_graph, float* d_pheroneme, float* d_probability, int* d_solutions)
{
  //1) generate a solution (haithem)
  //2) update the pheroneme based on the solution(mohamed)
  int tid = threadIdx.x;

}
  

/*
 * Main program and benchmarking 
 */
int main(int argc, char** argv)
{


  // allocate host memory 
  unsigned int nb_node   = GRAPH_SIZE; 
  unsigned int size      = GRAPH_SIZE*GRAPH_SIZE;             
  unsigned int mem_size  = sizeof(int) * size;     
  int*   h_graph         = (int*)malloc(mem_size); 
  float* h_pheroneme     = (float*)malloc(mem_size);
  float* h_probability   = (float*)malloc(mem_size);
  int*   h_solutions     = (int*)malloc(mem_size); 

  printf("Input size : %d\n", GRAPH_SIZE);
  printf("Grid size  : %d\n", GRID_SIZE);
  printf("Block size : %d\n", BLOCK_SIZE);

  //Initialise the graph, the pheroneme and the probabilities
  datainit_graph(h_graph, nb_node);
  datainit_pheroneme(h_pheroneme, nb_node);
  float* sum =sum_probability(h_graph, h_pheroneme, nb_node);
  update_probability(h_graph, h_pheroneme, h_probability, nb_node, sum);
  
  // allocate device memory
  int* d_graph;
  cutilSafeCall(cudaMalloc((void**) &d_graph, mem_size));
  float* d_pheroneme;
  cutilSafeCall(cudaMalloc((void**) &d_pheroneme, mem_size));
  float* d_probability;
  cutilSafeCall(cudaMalloc((void**) &d_probability, mem_size));
  int* d_solutions;
  cutilSafeCall(cudaMalloc((void**) &d_solutions, mem_size));  
  

  // copy host memory to device

  //The graph needs to be copied in the constant memory!!!!!!!!!!!!!!!
  cutilSafeCall(cudaMemcpy(d_graph, h_graph, 
				      mem_size, cudaMemcpyHostToDevice));

  cutilSafeCall(cudaMemcpy(d_pheroneme, h_pheroneme, 
				      mem_size, cudaMemcpyHostToDevice));

   cutilSafeCall(cudaMemcpy(d_probability, h_probability, 
              mem_size, cudaMemcpyHostToDevice));             

  // set up kernel for execution
  printf("Run %d Kernels.\n\n", ITER_BENCHMARK);
  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));
  cutilCheckError(cutStartTimer(timer));  


// execute kernel
  for (int j = 0; j < ITER_BENCHMARK; j++) 
      ACO_kernel<<<GRID_SIZE, BLOCK_SIZE >>>(d_graph, d_pheroneme, d_probability, d_solutions);

  // check if kernel execution generated and error
  cutilCheckMsg("Kernel execution failed");

  // wait for device to finish
  cudaThreadSynchronize();

  // stop and destroy timer
  cutilCheckError(cutStopTimer(timer));
  double dSeconds = cutGetTimerValue(timer)/(1000.0);
  double dNumOps = ITER_BENCHMARK * size;
  double gflops = dNumOps/dSeconds/1.0e9;

  //Log througput
  printf("Throughput = %.4f GFlop/s\n", gflops);
  cutilCheckError(cutDeleteTimer(timer));

  // copy result from device to host
  cutilSafeCall(cudaMemcpy(h_solutions, d_solutions, 
				       mem_size, cudaMemcpyDeviceToHost));

  // clean up memory
  free(h_graph);
  free(h_pheroneme);
  free(h_probability);
  free(h_solutions);
  cutilSafeCall(cudaFree(d_graph));
  cutilSafeCall(cudaFree(d_pheroneme));
  cutilSafeCall(cudaFree(d_probability));
  cutilSafeCall(cudaFree(d_solutions));

  // exit and clean up device status
  cudaThreadExit();
}

// 
void datainit_graph(int* h_graph, int size)
{    
    //same method as the CPU version
    int i,j,index;
    for(i=0 ; i<size ; i++)
    {
        for(j=0 ; j<size ; j++)
        {
            index = SERIALIZE(i,j,size);

            if(i < j) {

                h_graph[index] = 1;
            }
            else {
            h_graph[index] = 0;
          }
        }
    }
 
}

void datainit_pheroneme(float* h_pheroneme, int size)
{
  //same method as the CPU version
    int i,j,index;
    for(i=0 ; i<size ; i++)
    {
        for(j=0 ; j<size ; j++)
        {
            index = SERIALIZE(i,j,size);
            if(i < j)
                h_pheroneme[index] = INIT_PHERONEME;
            else{
            h_pheroneme[index] = 0;}
        }
    }

}

void datainit_graph_cube(int *graph,int max_depth) {
    
    //calculate the number of nodes available
    long num_nodes = max_cube_moves(max_depth);

    //calculate the number of nodes that are at depth max_depth -1
    long num_nodes_at_depth_minus_one = max_cube_moves(max_depth - 1);
    int i, j;
    int k =0;
    for (i=0 ; i < num_nodes_at_depth_minus_one; i++) {
      for (j=0 ; j < num_nodes; j++) {

        if ( ( j >= i * 18 + 1  && j< i*18 +1+k ) || ( j>=i*18+1+k+3 && j < (i * 18 + 18 + 1) ) ) {
          graph[i * num_nodes + j] = 1;
        }
        else {
          graph[i * num_nodes + j] = 0;
         }
      }
	if (i != 0 && i%3==0) k+=3;
    }

    //put zeros in the last level of nodes
    for (i=num_nodes_at_depth_minus_one ; i < num_nodes; i++) {
      for (j=0 ; j < num_nodes; j++) {
          graph[i * num_nodes + j] = 0;
      }
    }
}

/*
void datainit_graph_cube(int *graph,int max_depth) {
    
    //calculate the number of nodes available
    long num_nodes = max_cube_moves(max_depth);

    //calculate the number of nodes that are at depth max_depth -1
    long num_nodes_at_depth_minus_one = max_cube_moves(max_depth - 1);
    int i;

    for (i=0 ; i < num_nodes_at_depth_minus_one; i++) {
      if ( i >= i * 18 + 1 && i < (i * 18 + 18) ) {
        graph[i] = 1;
      }
      else {
        graph[i] = 0;
       }
    }

    //put zeros in the last level of nodes
    for (i=num_nodes_at_depth_minus_one ; i < num_nodes; i++) {
      graph[i] = 0;
    }
    int i,j;
    //start from node 2, and keep track of the next node number
    int current_node = 2;


    //initialize the first row
    for (i =0; i < num_nodes; i++) {

          int index = SERIALIZE(i,j,18);
            graph[index] = 0;
            current_node++:      
    }


    for (i =0; i < num_nodes; i++) {

        for (j=0; j<18; j++) {

          int index = SERIALIZE(i,j,18);

          if () { 
            graph[index] = current_node;
            current_node++:
          }
          else {
            graph[index] = 0;
          }

        }
    }   
}
*/



void update_pheroneme(float* h_pheroneme, int size)
{
    int i,j,index;
    //evaporation
    for(i=0 ; i<size ; i++)
    {
        for(j=0 ; j<size ; j++)
        {
            index = SERIALIZE(i,j,size);
            if(h_pheroneme[index] != 0)
            {
                h_pheroneme[index] = (1-EVAP_RATE) * h_pheroneme[index];
            }
        }
    }
}


float* sum_probability(int* h_graph, float* h_pheroneme, int size)

{
    int i,j,index;
    float* sum = (float*)malloc(sizeof(float)*size);
    for(i=0 ; i<size ; i++)
    {
        sum[i]=0;
        for(j=0 ; j<size ; j++)
        {
            index = SERIALIZE(i,j,size);
            if(h_graph[index] != 0 && h_pheroneme[index] != 0){
                sum[i] += pow(h_pheroneme[index],ALPHA) * pow(1/h_graph[index],BETA);
            }
        }
    }
    return sum;
}


void update_probability(int* h_graph,float* h_pheroneme,float* h_probability, int size, float* sum)
{
    //same methode as the CPU version
    int i,j,index;
    for(i=0 ; i<size ; i++)
    {
        for(j=0 ; j<size ; j++)
        {
            index = SERIALIZE(i,j,size);
            if(h_graph[index] != 0 && h_pheroneme[index] != 0)
            {
                h_probability[index] = pow(h_pheroneme[index],ALPHA) * pow(1/h_graph[index],BETA)/sum[i];
            }
            else{
                h_probability[index] = 0;
            }
        }
    }

}

