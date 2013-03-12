


//function that initialize the pheroneme to a constant value
void init_pheroneme(float * T, int size)
{
    int i,j,index;
    for(i=0 ; i<size ; i++)
    {
        for(j=0 ; j<size ; j++)
        {
            index = size*i + j;
            if(i < j)
                T[index] = C;
            else{
            T[index] = 0;}
        }
    }
}



void update_pheroneme1(float * T, int size, int * sol, int length_sol)
{

    //update based on constructed solution
    int i=0;
    while(sol[i] != size-1)
    {
        T[size*sol[i] + sol[i+1]] += Q/length_sol;
        i++;
    }

}

void update_pheroneme2(float * T, int size)
{
    int i,j,index;

    //evaporation
    for(i=0 ; i<size ; i++)
    {
        for(j=0 ; j<size ; j++)
        {
            index = size*i + j;
            if(i < j)
            {
                T[index] = (1-p) * T[index];
            }
        }
    }
}

