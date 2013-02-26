#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//
#define ITER_MAX 50
//evaporation rate
#define p  0.5
//influence rate of the pheroneme
#define alpha 0.6
//influence rate of the distance
#define beta 0.4

void init_graph(int * G, int size)
{
    int i,j;
    for(j=0 ; j<size ; j++)
    {
        for(i=0 ; i<size ; i++)
        {
            if(i==j)
                G[i+size*j] = 0;
            else{
            G[i+size*j] = 1;}
        }
    }
}

void init(float * data, int size)
{
    int i,j;
    for(j=0 ; j<size ; j++)
    {
        for(i=0 ; i<size ; i++)
        {
            data[i+size*j] = 0;
        }
    }
}

void update_pheroneme(float * T, int size)
{
    //evaporation
    int i,j;
    for(j=0 ; j<size ; j++)
    {
        for(i=0 ; i<size ; i++)
        {
            T[i+size*j] = (1-p) * T[i+size*j];
        }
    }

    //update based on constructed solutions

}

void update_prob(int * G, float * T, float * P, int size, float sum)
{
    int i,j;
    for(j=0 ; j<size ; j++)
    {
        for(i=0 ; i<size ; i++)
        {
            int index = i+size*j;
            P[index] = pow(T[index],alpha) * pow(G[index],beta)/sum;
        }
    }
}

float sum_prob(int * G, float * T, int size)
{
    int i,j;
    float sum = 0;
    for(j=0 ; j<size ; j++)
    {
        for(i=0 ; i<size ; i++)
        {
            int index = i+size*j;
            sum += pow(T[index],alpha) * pow(G[index],beta);
        }
    }
    return sum;
}

int main()
{
    //Let's denote n the number of nodes
    //and N the size of the problem
    int n = 5;
    int N = n*n;
    //We define the graph using a matrix of size N
    int * G = malloc(sizeof(int)*N);

    //We define the level of pheroneme for each edge in a matrix
    float * T = malloc(sizeof(float)*N);

    //We define the matrice of the probabilities
    float * P = malloc(sizeof(float)*N);

    //initialize the three matrices
    init_graph(G,n);
    init(T,n);
    init(P,n);

    //Let's define the number of ants that are going to go through the graph
    int nb_ants = 10;

    //number of iteration
    int iter = 0;
    while(iter<ITER_MAX)
    {
        //for each ant construct a path from the starting point to the final point
        int k;
        for(k=0 ; k<nb_ants ; k++)
        {

        }

        //update pheroneme values
        update_pheroneme(T,n);

        //update probabilities
        float sum = sum_prob(G,T,n);
        update_prob(G,T,P,n,sum);

        //increment iters
        iters += 1;
    }

    //Free the memory
    free(G);
    free(T);
    free(P);

    return 0;
}
