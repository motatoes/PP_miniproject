#include "stdio.h"
#include "stdlib.h"
#include "math.h"


//some constants
//6 faces ..
const int FACES = 6;

//18 moves (6 faces, and each face has three moves (clockwise, counterclockwise and 180)
const int NMOVES = 18;

//number of cubies for both corners and edges: corners: 8 positions x 3 orientations - edges: 12 positions x 2 orientations
const int CUBIES = 24;

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





struct CubePos {	
	//representing the corners
	//each cubie is represnted by 5 bits in the character
	//
	unsigned char c[8];

	//representing the edges
	unsigned char e[12];
	
	//the faces
};


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
static int corner_ori_add(int cv1, int cv2)  { return cv1 + cv2 % 3 ; }
static int corner_ori_sub(int cv1, int cv2)  { return cv1 - cv2 % 3 ;}

//we initialize the corners and edges with zero orientation, which is true for a 
//soved cube

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

//we need a lookup table that is going to tell us about how the orientations and the permutations are going to move
//for each of the 18 moves that we are going to perform. So we have 2 tables for our corners and permutations
//and these tables are of size (NMOVES X CUBIES) [18 X 24] for both corners and eges.

//however, we decide to serialize the array because it makes it clear which locations are contigous,we do this row by row:

static unsigned char edge_trans   [NMOVES * CUBIES];
static unsigned char corner_trans [NMOVES * CUBIES];


//"Corner permutation. We can do the same thing for the corner permutation. A quarter twist of the U
//face moves the corner in slot 0 to slot 1, from slot 1 to slot 3, from slot 3 to slot 2, and from slot 2 to slot 0.
//This permutation is (0; 1; 3; 2), and it's the first entry in the array below. This array is carefully constructed
//so the first two slots are always from the U face (assuming any slots are), which simplfies some later code."
//
//however, we decide to serialize the array because it makes it clear which locations are contigous,we do this row by row:


static const unsigned char corner twist perm[FACES * 4] = {
							    0, 1, 3, 2, 
							    2, 3, 7, 6,
							    3, 1, 5, 7, 
							    4, 6, 7, 5,
							    1, 0, 4, 5,
							    0, 2, 6, 4
							  };

//"Edge orientation convention. Now we consider the orientation aspects of moves. When we say a
//corner is twisted, or an edge is 
//flipped, that makes sense when the cubie is in its solved position. But what
//does it mean for a cubie to be twisted or 
//flipped when it is in some other slot?
//Let us start by considering edge 
//flip. Consider the edge cubie whose home location is the intersection of
//the U and F faces (we can call this cubie UF). If we permit only the moves U, F, D, and B (and half-twists
//and counterclockwise twists), it is straightforward to see that whenever the cubie UF is in the U or D face,
//its U facelet (the sticker colored the same as the center cubie on the U face) is always on the U or D face,
//and never on one of the F, R, B, or L faces. Further, when the UF cubie is in the middle layer, its U facelet
//is always on the L or R face. In other words, there is only a single orientation for each cubie in each slot if
//we start from the solved position and perform only combinations of the moves U, F, D, and B.
//If we further permit R and L moves, however, this is no longer true. In particular, the move sequence
//F1R1U1 brings the UF cubie back to the UF slot, but now the U facelet is in the front face.
//We can thus define an edge orientation convention as follows. Only the four moves R1, R3, L1, and L3
//modify the edge orientation of any cubie as the cubie moves along slots. All other moves preserve the edge
//orientation."

static const unsigned char edge_change[FACES] = {0, 0, 1, 0, 0, 1};


//"Corner orientation convention.
//Corner orientation is similar, but there are three possible orientations for every cubie, not just two. Note
//that every cubie has a U or D facelet; this permits a straightforward orientation convention based on simple
//examination. If the U or D facelet is in the U or D face, we declare the cubie to be properly oriented
//(an orientation of 0). If twisting the cubie (when looking towards the center of the cube from that cubie)
//counterclockwise brings it into the oriented state, then we consider the cubie to be oriented clockwise, or
//+1. If twisting the cubie clockwise brings it into the oriented state, we consider the cubie to be oriented
//counterclockwise, or +2 (which is ? ? 1 mod 3).
//From this denition, it is clear that no move of the U or D faces will change the orientation of any corner
//cubie. A quarter twist of any other face that leaves a particular corner cubie in the same U or D face that it
//started from will ect a clockwise twist on that cubie. A quarter twist that moves a corner cube from the U
//face to the D face, or from the D face to the U face, will ect a counterclockwise twist on that cubie. This
//can be summarized in the following array. Note that we use the information that the corner twist perm array
//above always starts with two U face slots before listing two D face slots; thus, the transition corresponding
//to elements 0 and 2 preserve the U or D face of a cubie, while the elements for 1 and 3 move a cubie from
//the U face to the D face or vice versa."

//however, we decide to serialize the array because it makes it clear which locations are contigous,we do this row by row:
static const unsigned char corner change [FACES * 4] = {
							   0, 0, 0, 0,
							   1, 2, 1, 2,
							   1, 2, 1, 2,
							   0, 0, 0, 0,
							   1, 2, 1, 2,
							   1, 2, 1, 2 
			 			       };
 


int main() {
	struct Cube base;
	cube_init(base);
}
