#include "stdio.h"
#include "stdlib.h"
#include "math.h"

/************************************************************************************************************
*																											*
* most of the concepts and methods in this file have been adapted from: http://cube20.org/src/cubepos.pdf	*
*																											*
*																											*
*************************************************************************************************************/



//the basic definition of corners and edges is defined here.


//some constants
//6 faces ..
#define FACES  6

//18 moves (6 faces, and each face has three moves (clockwise, counterclockwise and 180
#define NMOVES  18

//number of cubies for both corners and edges: corners: 8 positions x 3 orientations - edges: 12 positions x 2 orientations
#define CUBIES  24

//the number of twists
#define TWISTS 3




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





typedef struct CubePos {	
	//representing the corners
	//each cubie is represnted by 5 bits in the character
	//
	unsigned char c[8];

	//representing the edges
	unsigned char e[12];
	
	//the faces
} CubePos;


//these are some useful utility methods to pull out information about the permutation
//and orientation from a specific position of the cube. We use them throughout the 
//program


static int edge_perm(int cubieval) 	     { return cubieval >> 1 ; }
static int edge_ori(int cubieval) 	     { return cubieval & 1 ; }
static int corner_perm(int cubieval) 	     { return cubieval & 7 ; }
static int corner_ori(int cubieval)	     { return cubieval >> 3 ; }
static int edge_flip(int cubieval) 	     { return cubieval ^ 1 ; }
static int edge_val(int perm, int ori) 	     { return perm * 2 + ori ; }
static int corner_val(int perm, int ori)     { return ori * 8 + perm ; }
static int edge_ori_add(int cv1, int cv2)    { return cv1 ^ edge_ori(cv2) ; }
static int corner_ori_add(int cv1, int cv2)  { return (cv1 + cv2 & 0x18) % 24 ; }
static int corner_ori_sub(int cv1, int cv2)  { return cv1 + corner_val(0, (3 - corner_ori(cv2) ) % 3) ;}

//we initialize the corners and edges with zero orientation, which is true for a 
//soved cube

void cube_init(CubePos* b) {
   int i;
   for (i=0; i<8; i++)
   		(*b).c[i] = (char)corner_val(i, 0) ;
   for (i=0; i<12; i++)
   		(*b).e[i] = (char)edge_val(i, 0) ;
}


void print_corners_and_edges(CubePos *c) {
	int i;
	printf("Corners --\n");
	for (i=0;i<8;i++)
		printf("%d ", (*c).c[i]);

	printf("\nEdges -- \n ");

	for (i=0;i<12;i++)
		printf(" %d ", (*c).e[i]);

	printf("\n\n ");

}

