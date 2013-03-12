void update_prob(int * G, float * T, float * P, int size, float * sum)
{
    int i,j,index;
    for(i=0 ; i<size ; i++)
    {
        for(j=0 ; j<size ; j++)
        {
            index = size*i + j;
            if(G[index] != 0 && T[index] != 0)
            {
                P[index] = pow(T[index],ALPHA) * pow(1/G[index],BETA)/sum[i];
            }
            else{
                P[index] = 0;
            }
        }
    }
}

float * sum_prob(int * G, float * T, int size)
{
    int i,j,index;
    float * sum = malloc(sizeof(float)*size);
    for(i=0 ; i<size ; i++)
    {
        sum[i]=0;
        for(j=0 ; j<size ; j++)
        {
            index = size*i + j;
            if(G[index] != 0 && T[index] != 0){
                sum[i] += pow(T[index],ALPHA) * pow(1/G[index],BETA);
            }
        }
    }
    return sum;
}

