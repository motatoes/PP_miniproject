#include "stdio.h"
#include "stdlib.h"
#include "math.h"

//rubik's cube libs
#include "parse.c"
#include "move.c"


int main() {

   INIT_CUBE_MOVES();
   INIT_CUBE_PARSE();

   CubePos base;
   cube_init(&base);
   print_corners_and_edges(&base);
   //move(&base, RIGHT);
   move(&base, RIGHT);
   print_corners_and_edges(&base);

   char *sm = "UF UR UB UL DF DR DB DL FR FL BR BL UFR URB UBL ULF DRF DFL DLB DBR ";
   const char *r;
   r = "asdas";



   //r = parse_Singmaster(&base, sm);

   char *a = Singmaster_string(&base);

   printf("%s", a);

}