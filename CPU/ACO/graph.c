

void init_graph(int * G, int size)
{
    int i,j,index;
    for(i=0 ; i<size ; i++)
    {
        for(j=0 ; j<size ; j++)
        {
            index = size*i + j;
            if(i < j)
                G[index] = 1;
            else{
            G[index] = 0;}
        }
    }
}


