#include "stdio.h"
#include "stdlib.h"
#include "math.h"


//some constants
#define FACES 6

//most of the concepts and methods in the file have been adapted from: http://cube20.org/src/cubepos.pdf

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



char** parseCube(char* p) {

}



//internally, we represent the cube as 8 corners, and 12 edges. These are presented in their 
//order of appearance in three slices in the cube

//
//					** 12 edges**
//			    ___   ___   ___      ___   ___   ___      ___   ___   ___
//			   |___| |_0_| |___|    |_4_| |___| |_5_|    |___| |_8_| |___|  
//			    ___   ___   ___      ___   ___   ___      ___   ___   ___   
//			   |_1_| |___| |_2_|    |___| |___| |___|    |_9_| |___| |_10|  
//			    ___   ___   ___      ___   ___   ___      ___   ___   ___  
//			   |___| |_3_| |___|    |_6_| |___| |_7_|    |___| |_11| |___|  

//					** 8 corners**
//			    ___   ___   ___      ___   ___   ___      ___   ___   ___
//			   |_0_| |___| |_1_|    |___| |___| |___|    |_4_| |___| |_5_|  
//			    ___   ___   ___      ___   ___   ___      ___   ___   ___   
//			   |___| |___| |___|    |___| |___| |___|    |___| |___| |___|  
//			    ___   ___   ___      ___   ___   ___      ___   ___   ___  
//			   |_2_| |___| |_3_|    |___| |___| |___|    |_6_| |___| |_7_|  


//corners are represented using 5 digits, these are split into two parts 
// 1) the cube permutation (as shown above) , P
// 2) the cube orientation (two orientations for an edge, three for a corner), O

//for the corner, we represent it as follows (three orientations, 8 perms):
//		O O P P P

//for the edge, we represent it as follows (two orientations, 12 perms):
//		P P P P O





struct Cube {	
	//representing the corners
	//each cubie is represnted by 5 bits in the character
	//
	unsigned char c[8];

	//representing the edges
	unsigned char e[12];
	
	//the faces
};




static int edge_perm(int cubieval) 			{ return cubieval >> 1 ; }
static int edge_ori(int cubieval) 			{ return cubieval & 1 ; }
static int corner_perm(int cubieval) 		{ return cubieval & 7 ; }
static int corner_ori(int cubieval)		 	{ return cubieval >> 3 ; }
static int edge_flip(int cubieval) 		    { return cubieval ^ 1 ; }
static int edge_val(int perm, int ori) 	    { return perm * 2 + ori ; }
static int corner_val(int perm, int ori)     { return ori * 8 + perm ; }
static int edge_ori_add(int cv1, int cv2)    { return cv1 ^ edge_ori(cv2) ; }
static int corner_ori_add(int cv1, int cv2)  { return cv1 + cv2 % 3 ; }
static int corner_ori_sub(int cv1, int cv2)  { return cv1 - cv2 % 3 ;}

void cube_init(struct Cube b) {
   int i;
   for (i=0; i<8; i++)
   		b.c[i] = (char)corner_val(i, 0) ;
   for (i=0; i<12; i++)
   		b.e[i] = (char)edge_val(i, 0) ;
}

//"The order we adopt is U, F, R, D, B, L, to which we assign the ordinals 0 to 5. This ordering has the
//following properties:
//1. The opposite face of face i is i + 3 mod FACES
//2. Every three cyclically consecutive faces in this ordering join at a single corner. This defines six of the
//eight corners; the other two are defined by the odd-numbered and the even-numbered faces, respectively.
//3. Every pair of faces whose ordinals do not differ by 3 (mod FACES) defines an edge."

const char faces[FACES] = {'U', 'F', 'R', 'D', 'B', 'L'};


int main() {
	struct Cube base;
	cube_init(base);
}
