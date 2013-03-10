#include "stdio.h"
#include "stdlib.h"
#include "math.h"

//prevent duplication when adding 'core.c'
#ifndef CORE_C
#define CORE_C
#include "core.c"
#endif


#define UP 0
#define UP2 1
#define UPA 2
#define FRONT 3
#define FRONT2 4
#define FRONTA 5
#define RIGHT 6
#define RIGHT2 7
#define RIGHTA 8
#define DOWN 9
#define DOWN2 10
#define DOWNA 11
#define BACK 12
#define BACK2 13
#define BACKA 14
#define LEFT  15 
#define LEFT2 16
#define LEFTA 17


//maco function to initialize the moves of the cube
#define INIT_CUBE_MOVES()  init_twist_table();permute_twist_table();


/************************************************************************************************************
*																											*
* most of the concepts and methods in this file have been adapted from: http://cube20.org/src/cubepos.pdf	*
*																											*
*																											*
*************************************************************************************************************/


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
//we call both of these 'twist-tables'
//however, we decide to serialize the array because it makes it clear which locations are contigous,we do this row by row:

unsigned char edge_trans   [NMOVES * CUBIES];
unsigned char corner_trans [NMOVES * CUBIES];


//we can also initialize these two twist-table with the initial state of all the corners, this is going to be filling in [0..23]
//for all the faces, which represents both corners and edges
//ofcourse, we need to change this to reflect changes to the corners/edges when a face rotation occurs, and we will do this
//in a bit
void init_twist_table() {
	int n,c;
	for (n = 0 ; n < NMOVES ; n++) {
		for (c=0 ; c < CUBIES ; c++) {
			edge_trans[n * CUBIES + c] = c;
			corner_trans[n * CUBIES + c] = c;
		}
	} 
}
//now, before we "fix" this table to define what happens when a face actually changes, we need to first define some data structures
//these data structures will be used to later determine what happens when a face-twist occurs (changes in permutation and orientation).  


// 1- "
//Edge permutation. We need to now fill in the edge trans and corner trans arrays. Based on our cubie
//numbering, we can build an array listing the slots that are acted by a clockwise twist of each face, in the
//order of the moves, based on our slot numbering convention and move numbering convention. A clockwise
//twist of the first face (U) moves the cubie from slot 0 into slot 2, from slot 2 into slot 3, from slot 3 into
//slot 1, and from slot 1 into slot 0. This is represented by the permutation written as (0,2,3,1) and this comprises
//the first element of the following array. The rest are filled in similarly"
//

static const unsigned char edge_twist_perm[FACES * 4] = {
								0, 2, 3, 1,
								3, 7, 11, 6,
								2, 5, 10, 7,
								9, 11, 10, 8,
								0, 4, 8, 5, 
								1, 6, 9, 4
							};

// 2- "Corner permutation. We can do the same thing for the corner permutation. A quarter twist of the U
//face moves the corner in slot 0 to slot 1, from slot 1 to slot 3, from slot 3 to slot 2, and from slot 2 to slot 0.
//This permutation is (0; 1; 3; 2), and it's the first entry in the array below. This array is carefully constructed
//so the first two slots are always from the U face (assuming any slots are), which simplfies some later code."
//
//however, we decide to serialize the array because it makes it clear which locations are //contigous,we do this row by row:


static const unsigned char corner_twist_perm[FACES * 4] = {
							    0, 1, 3, 2, 
							    2, 3, 7, 6,
							    3, 1, 5, 7, 
							    4, 6, 7, 5,
							    1, 0, 4, 5,
							    0, 2, 6, 4
							  };

// 3- "Edge orientation convention. Now we consider the orientation aspects of moves. When we say a
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


// 4 -"Corner orientation convention.
//Corner orientation is similar, but there are three possible orientations for every cubie, not just two. Note
//that every cubie has a U or D facelet; this permits a straightforward orientation convention based on simple
//examination. If the U or D facelet is in the U or D face, we declare the cubie to be properly oriented
//(an orientation of 0). If twisting the cubie (when looking towards the center of the cube from that cubie)
//counterclockwise brings it into the oriented state, then we consider the cubie to be oriented clockwise, or
//+1. If twisting the cubie clockwise brings it into the oriented state, we consider the cubie to be oriented
//counterclockwise, or +2 (which is - - 1 mod 3).
//From this definition, it is clear that no move of the U or D faces will change the orientation of any corner
//cubie. A quarter twist of any other face that leaves a particular corner cubie in the same U or D face that it
//started from will ect a clockwise twist on that cubie. A quarter twist that moves a corner cube from the U
//face to the D face, or from the D face to the U face, will effect a counterclockwise twist on that cubie. This
//can be summarized in the following array. Note that we use the information that the corner twist perm array
//above always starts with two U face slots before listing two D face slots; thus, the transition corresponding
//to elements 0 and 2 preserve the U or D face of a cubie, while the elements for 1 and 3 move a cubie from
//the U face to the D face or vice versa."

//however, we decide to serialize the array because it makes it clear which locations are contigous,we do this row by row:
static const unsigned char corner_change [FACES * 4] = {
							   0, 0, 0, 0,
							   1, 2, 1, 2,
							   1, 2, 1, 2,
							   0, 0, 0, 0,
							   1, 2, 1, 2,
							   1, 2, 1, 2 
			 			       };
 

//now we need to modify the twist-table to what happens when each move occurs. We know that the rows of the twist turn
//represent actual "moves" starting from U,U2,U'... up until L,L2,L' . We also know that each twist results in a 
//permutation of 4 corners and 4 edges. This is what we defined earlier. We also know which twists cause an orientaion to 
//change, and which don't. In general, half-turn twists will not result in a change of orientation (for both corners and
//edges).
//
//For the edge twist-table, we notice that for each row (twist-turn), 8 columns are affected -- 4 edges, and each edge can have 2 permutations
//
//For the corner twist-table, we notice that for each row (twist-turn), 12 columns are affected -- 4 corners, and each corner can have 3 permutations
//
//It will get clearer after you see the move function after this one
//Now let's see the code
//

void permute_twist_table() {
	int f,t,i,o;
	for (f = 0; f < FACES; f ++)
		for (t = 0; t < 3; t++) {
			int m = f * TWISTS + t;
			int isquarter = (t == 0 || t == 2);
			int perminc = t + 1;
			if (m < 0) continue;
			for (i = 0; i < 4; i++) {
				int ii = (i + perminc) % 4;
				for (o = 0; o < 2; o++) {
					int oo = o; /* new orientation */
					if (isquarter) oo ^= edge_change[f]; //bitwise xor (if edge_change=1, oo=¬oo, otherwise oo=oo) 
					edge_trans[m * CUBIES + edge_val (edge_twist_perm[f * 4 +i], o)] = edge_val(edge_twist_perm[f * 4 + ii ], oo);
				}
				for (o = 0; o < 3; o++) {
					int oo = o; /* new orientation */
					if (isquarter) oo = (corner_change [f * 4 + i] + oo) % 3;
					corner_trans [m * CUBIES + corner_val (corner_twist_perm[f * 4 + i], o)] = corner_val(corner_twist_perm[f * 4 + ii ], oo);
				}
			}
		}
}
//we can now (finally) introduce our move function that performs a twist operation. 
//The function takes an integer that represents the index of the move (similar to the order described above)  
//and the cube to manipulate, and manipulates it



void move(CubePos *C, int mov) {
	const unsigned char *p = &corner_trans[mov * CUBIES];
	(*C).c[0] = p[(*C).c[0]];
	(*C).c[1] = p[(*C).c[1]];
	(*C).c[2] = p[(*C).c[2]];
	(*C).c[3] = p[(*C).c[3]];
	(*C).c[4] = p[(*C).c[4]];
	(*C).c[5] = p[(*C).c[5]];
	(*C).c[6] = p[(*C).c[6]];
	(*C).c[7] = p[(*C).c[7]];
	 p = &edge_trans[mov * CUBIES];
	(*C).e[0] = p[ (*C).e[0]];
	(*C).e[1] = p[ (*C).e[1]];
	(*C).e[2] = p[ (*C).e[2]];
	(*C).e[3] = p[ (*C).e[3]];
	(*C).e[4] = p[ (*C).e[4]];
	(*C).e[5] = p[ (*C).e[5]];
	(*C).e[6] = p[ (*C).e[6]];
	(*C).e[7] = p[ (*C).e[7]];
	(*C).e[8] = p[ (*C).e[8]];
	(*C).e[9] = p[ (*C).e[9]];
	(*C).e[10] = p[ (*C).e[10]];
	(*C).e[11] = p[ (*C).e[11]];
}




