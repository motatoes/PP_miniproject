
void print_int(int * data, int size)
{
    int i,j,index;
    for(i=0 ; i<size ; i++)
    {
        for(j=0 ; j<size ; j++)
        {
            index = size*i + j;
            printf("%d ",data[index]);
        }
        printf("\n");
    }
}

void print_float(float * data, int size)
{
    int i,j,index;
    for(i=0 ; i<size ; i++)
    {
        for(j=0 ; j<size ; j++)
        {
            index = size*i + j;
            printf("%f ",data[index]);
        }
        printf("\n");
    }
}