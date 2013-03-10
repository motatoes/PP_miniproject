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
   //print_corners_and_edges(&base);
   

  for (i=1; i< argc; i++) {
    move(&base, (int)atoi(argv[i]));
}
  // print_corners_and_edges(&base);

   char *sm = "UF UR UB UL DF DR DB DL FR FL BR BL UFR URB UBL ULF DRF DFL DLB DBR ";
   const char *r;
   
   //r = parse_Singmaster(&base, sm);

   char *a = Singmaster_string(&base);

   printf("%s", a);

}
