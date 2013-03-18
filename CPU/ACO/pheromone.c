


//function that initialize the pheromone to a constant value
void init_pheromone(float * T, int size)
{
    int i,j,index;
    for(i=0 ; i<size ; i++)
    {
        for(j=0 ; j<size ; j++)
        {
            index = size*i + j;
            if(i < j)
                T[index] = INIT_PHEROMONE;
            else{
            T[index] = 0;}
        }
    }
}



void update_pheromone1(float * T, int size, int * sol, int length_sol)
{

    //update based on constructed solution
    int i=0;
    while(sol[i] != size-1)
    {
        T[size*sol[i] + sol[i+1]] += UPDT_PHEROMONE_CONST/length_sol;
        i++;
    }

}

void update_pheromone2(float * T, int size)
{
    int i,j,index;

    //evaporation
    for(i=0 ; i<size ; i++)
    {
        for(j=0 ; j<size ; j++)
        {
            index = size*i + j;
            if(T[index] != 0)
            {
                T[index] = (1-EVAP_RATE) * T[index];
            }
        }
    }
}

