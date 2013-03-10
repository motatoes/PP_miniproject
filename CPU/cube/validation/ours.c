#include "stdio.h"
#include "stdlib.h"
#include "math.h"

//rubik's cube libs
#include "../parse.c"
#include "../move.c"


int main(int argc,char ** argv) {
	
   
   INIT_CUBE_MOVES();
   INIT_CUBE_PARSE();

int i;

   CubePos base;
   cube_init(&base);

  for (i=1; i< argc; i++) {
    move(&base, (int)atoi(argv[i]));
}
   //"UF UR UB UL DF DR DB DL FR FL BR BL UFR URB UBL ULF DRF DFL DLB DBR ";
   char *a = Singmaster_string(&base);

   printf("%s", a);

}
