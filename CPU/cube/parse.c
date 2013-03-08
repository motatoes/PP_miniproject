
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h" /* memset */

//prevent duplication when adding 'core.c'
#ifndef CORE_C
#define CORE_C
#include "core.c"
#endif


//maco function to initialize the singmaster parsing of the cube
#define INIT_CUBE_PARSE() sm_init();


//this file is about parsing a string in singmaster's notation and ouptutting a 
//string that's reformatted to the internal representation of the cube

//the program takes as input the cube represented in singmaster's notation:
//	        						    ___   ___   ___ 
//								   |_U_| |_U_| |_U_|
//								    ___   ___   ___ 
//								   |_U_| |___| |_U_|
//								    ___   ___   ___ 
//								   |_U_| |_U_| |_U_|
//			       ___   ___   ___   ___   ___   ___    ___   ___   ___   ___   ___   ___    
//	      		      |_L_| |_L_| |_L_| |_F_| |_F_| |_F_|  |_R_| |_R_| |_R_| |_B_| |_B_| |_B_|  
//			       ___   ___   ___   ___   ___   ___    ___   ___   ___   ___   ___   ___   
//			      |_L_| |___| |_L_| |_F_| |___| |_F_|  |_R_| |___| |_R_| |_B_| |___| |_B_|  
//			       ___   ___   ___   ___   ___   ___    ___   ___   ___   ___   ___   ___ 
//		 	      |_L_| |_L_| |_L_| |_F_| |_F_| |_F_|  |_R_| |_R_| |_R_| |_B_| |_B_| |_B_|  
//   		    				                    ___   ___   ___ 
//			 				           |_D_| |_D_| |_D_|
//								    ___   ___   ___ 
//							           |_D_| |___| |_D_|
//							            ___   ___   ___  
//							           |_D_| |_D_| |_D_|



//corners are represented using the three faces that they meet in a solved cube
//  edges are represented using the two   faces that they meet in a solved cube
//



static char static_buf[200] ;


static const char *sing_solved =
"UF UR UB UL DF DR DB DL FR FL BR BL UFR URB UBL ULF DRF DFL DLB DBR" ;

static const char *const smedges[] = {
   "UB", "BU", "UL", "LU", "UR", "RU", "UF", "FU",
   "LB", "BL", "RB", "BR", "LF", "FL", "RF", "FR",
   "DB", "BD", "DL", "LD", "DR", "RD", "DF", "FD",
} ;
static const char *const smcorners[] = {
   "UBL", "URB", "ULF", "UFR", "DLB", "DBR", "DFL", "DRF",
   "LUB", "BUR", "FUL", "RUF", "BDL", "RDB", "LDF", "FDR",
   "BLU", "RBU", "LFU", "FRU", "LBD", "BRD", "FLD", "RFD",
   "ULB", "UBR", "UFL", "URF", "DBL", "DRB", "DLF", "DFR",
   "LBU", "BRU", "FLU", "RFU", "BLD", "RBD", "LFD", "FRD",
   "BUL", "RUB", "LUF", "FUR", "LDB", "BDR", "FDL", "RDF",
} ;


const int INVALID = 99 ;
static unsigned char lookup_edge_cubie[FACES*FACES] ;
static unsigned char lookup_corner_cubie[FACES*FACES*FACES] ;
static unsigned char sm_corner_order[8] ;
static unsigned char sm_edge_order[12] ;
static unsigned char sm_edge_flipped[12] ;

///////////////

void skip_whitespace(const char **p)   {
   while (**p && **p <= ' ')
   {
      (*p)++ ;
    }
}

int parse_face(const char **p) {
   int f = parse_face_from_char(*p) ;
   if (f >= 0)
      (*p)++ ;
   return f ; 
}
int parse_face_from_char(const char *f) {


   switch (*f) {
case 'u': case 'U': return 0 ;
case 'f': case 'F': return 1 ;
case 'r': case 'R': return 2 ;
case 'd': case 'D': return 3 ;
case 'b': case 'B': return 4 ;
case 'l': case 'L': return 5 ;
default:
      return -1 ;
   }
}
int parse_move(const char *p) {
   skip_whitespace(&p) ;
   const char *q = p ;
   int f = parse_face(&q) ;
   if (f < 0)
      return -1 ;
   int t = 0 ;
   switch (*q) {
case '1': case '+': t = 0 ; break ;
case '2': t = 1 ; break ;
case '3': case '\'': case '-': t = TWISTS-1 ; break ;
default:
      return -1 ;
   }
   p = q + 1 ;
   return f * TWISTS + t ;
}
/////////////////
static int parse_cubie(const char **p) {
   skip_whitespace(&*p) ;
   int v = 1 ;
   int f = 0 ;
   while ((f=parse_face(&*p)) >= 0) {
      v = v * 6 + f ;
      if (v >= 2 * 6 * 6 * 6)
         return -1 ;
   }
   return v ;
}
static int parse_edge(const char **p) {
   int c = parse_cubie(&*p) ;
   if (c < 6 * 6 || c >= 2 * 6 * 6)
      return -1 ;
   c = lookup_edge_cubie[c-6*6] ;
   if (c == INVALID)
      return -1 ;
   return c ;
}
static int parse_corner(const char **p) {
   int c = parse_cubie(&*p) ;
   if (c < 6 * 6 * 6 || c >= 2 * 6 * 6 * 6)
      return -1 ;
   c = lookup_corner_cubie[c-6*6*6] ;
   if (c == INVALID || c >= CUBIES)
      return -1 ;
   return c ;
}



//We need to initialize all of those arrays.

void sm_init() {

   int i;
   memset(lookup_edge_cubie, INVALID, sizeof(lookup_edge_cubie)) ;
   memset(lookup_corner_cubie, INVALID, sizeof(lookup_corner_cubie)) ;
   for (i=0; i<CUBIES; i++) {
      const char *tmp = 0 ;
      tmp=smcorners[i];
      lookup_corner_cubie[parse_cubie(&tmp)-6*6*6] = i ;
      tmp=smcorners[CUBIES+i];
      lookup_corner_cubie[parse_cubie(&tmp)-6*6*6] = CUBIES+i ;
      tmp=smedges[i];
      lookup_edge_cubie[parse_cubie(&tmp)-6*6] = i ;
   }
   const char *p = sing_solved ;
   for (i=0; i<12; i++) {
      int cv = parse_edge(&p) ;
      sm_edge_order[i] = edge_perm(cv) ;
      sm_edge_flipped[i] = edge_ori(cv) ;
   }
   for (i=0; i<8; i++)
      sm_corner_order[i] = corner_perm(parse_corner(&p)) ;
}

//inverting a corner and edge sequence
void invert_into(CubePos *old, CubePos *dst)  {
   int i;
   for (i=0; i<8; i++) {
      int cval = (*old).c[i] ;
      (*dst).c[corner_perm(cval)] = corner_ori_sub(i, cval) ;
   }
   for ( i=0; i<12; i++) {
      int cval = (*old).e[i] ;
      (*dst).e[edge_perm(cval)] = edge_val(i, edge_ori(cval)) ;
   }
}


const char *parse_Singmaster(CubePos *Result, const char *p) {
  if (strncmp(p, "SING ", 5) == 0)
      p+= 5;
   int m = 0;
   int i;
   for (i=0; i<12; i++) {
      int c = parse_edge(&p) ^ sm_edge_flipped[i];
       if (c < 0)
         return "No such edge";

      (*Result).e[edge_perm(c)] = edge_val(sm_edge_order[i], edge_ori(c));
      m |= 1<<(i);
   }
   for (i=0; i<8; i++) {
      int cval = parse_corner(&p);
      if (cval < 0)
         return "No such Corner";
      (*Result).c[corner_perm(cval)] =  corner_ori_sub(sm_corner_order[i], cval);
      m |= 1<<(i+12);
   }
   skip_whitespace(&p);
   if (*p) return "Extra stuff after Singmaster representation";
   if (m != ((1<<20) - 1)) return "Missing at least one cubie";
   return 0;

}


char *Singmaster_string(CubePos *toinvert)  {

   CubePos cp ;
   cube_init(&cp);
   int i;
   invert_into(toinvert, &cp) ;
   char *p = static_buf ;
   for (i=0; i<12; i++) {
      if (i != 0)
         *p++ = ' ' ;
      int j = sm_edge_order[i] ;
      const char *q = smedges[cp.e[j] ^ sm_edge_flipped[i]] ;
      *p++ = *q++ ;
      *p++ = *q++ ;
   }
   for (i=0; i<8; i++) {
      *p++ = ' ' ;
      int j = sm_corner_order[i] ;

      const char *q = smcorners[cp.c[j]] ;
      *p++ = *q++ ;
      *p++ = *q++ ;
      *p++ = *q++ ;
   }
   *p = 0 ;
   return static_buf ;
}



