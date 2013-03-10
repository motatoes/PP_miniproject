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
   CubePos test;
   cube_init(&base);
  
  for (i=1; i< argc; i++) {
	  move(&base, (int)atoi(argv[i]));
  }
   
   parse_Singmaster(&test, Singmaster_string(&base));
   int n = memcmp(&test,&base, 20);

   if ( n == 0 ) {
   	printf("1");

   }
   else {
	printf("0");

   }


}
