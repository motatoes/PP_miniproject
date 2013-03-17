/*
 * This program uses the host CURAND API to generate 100 
 * pseudorandom floats.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

int main(int argc, char *argv[])
{
    size_t n = 12;
    size_t i;
    curandGenerator_t gen;
    unsigned int *devData, *hostData;

    /* Allocate n floats on host */
    hostData = (unsigned int *)calloc(n, sizeof(unsigned int));

    /* Allocate n floats on device */
    CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(unsigned int)));

    /* Create pseudo-random number generator */
    CURAND_CALL(curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_DEFAULT));
    
    /* Set seed */
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 
                1234ULL));

    /* Generate n floats on device */
    CURAND_CALL(curandGenerate(gen, devData, n));

    /* Copy device memory to host */
    CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(unsigned int),
        cudaMemcpyDeviceToHost));

    /* Show result */
    for(i = 0; i < n; i++) {
        printf("%d ", hostData[i]);
    }
    printf("\n");

    /* Cleanup */
    CURAND_CALL(curandDestroyGenerator(gen));
    CUDA_CALL(cudaFree(devData));
    free(hostData);    
    return EXIT_SUCCESS;
}



void h_calculate_path_selections(unsigned int* path_selections, float* probabilities, int nb_ant, int graph_size) {

   int i,j,k,index;
   float rdm, node_probability, cummulative_probability;
    srand(time(NULL));
    
    //for each ant ...
    for (i=0; i<nb_ant; i++) {
    
         //givin that it was in node j ...
        for  (j=0; j<graph_size; j++) {
 
            rdm = rand()/(float)RAND_MAX;
            cummulative_probability = 0; 
            //which node will it chose , probabilistically?
            for(k=0; k<graph_size; k++)
            {
                
                index = SERIALIZE(j,k,graph_size);
                node_probability = probabilities[index];
                //this node (k) is unreachable from node j
                if (node_probability == 0) continue;
                
                printf("%f\n", rdm );
                cummulative_probability += node_probability;

                //if the random number is less or equal to
                //the probability to select the next node we select it
                if( rdm <= cummulative_probability )
                {
                    index = SERIALIZE(i,j,graph_size);
                    path_selections[index]=k;
                    break;
                }
            }


        }
    } 

}
