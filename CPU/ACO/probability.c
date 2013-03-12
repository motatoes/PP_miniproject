


void update_prob(int * G, float * T, float * P, int size, float * sum)
{
    int i,j,index;
    for(i=0 ; i<size ; i++)
    {
        for(j=0 ; j<size ; j++)
        {
            index = size*i + j;
            if(i < j)
            {
                P[index] = pow(T[index],alpha) * pow(1/G[index],beta)/sum[i];
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
            if(i < j){
                sum[i] += pow(T[index],alpha) * pow(1/G[index],beta);
            }
        }
    }
    return sum;
}


