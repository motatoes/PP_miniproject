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

//inverting a corner and edge sequence
void invert_into(cubepos &old, cubepos &dst) const {
   for (int i=0; i<8; i++) {
      int cval = old.c[i] ;
      dst.c[corner_perm(cval)] = corner_ori_sub(i, cval) ;
   }
   for (int i=0; i<12; i++) {
      int cval = old.e[i] ;
      dst.e[edge_perm(cval)] = edge_val(i, edge_ori(cval)) ;
   }
}


void skip_whitespace(const char *p)   {
   while (*p && *p <= ' ')
      p++ ;
}
int parse_face(const char *p) {
   int f = parse_face_from_char( *p) ;
   if (f >= 0)
      p++ ;
   return f ; 
}
int parse_face_from_char( char *f) {
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
   skip_whitespace(p) ;
   const char *q = p ;
   int f = parse_face(q) ;
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
void append_move(char **p, int mv) {
   append_face(p, mv/TWISTS) ;
   *p++ = "123"[mv % TWISTS] ;
   *p = 0 ;
}
moveseq parse_moveseq(const char **p) {
   moveseq r ;
   int mv ;
   while ((mv=parse_move(p)) >= 0)
      r.push_back(mv) ;
   return r ;
}
void append_moveseq(char **p, const moveseq *seq) {
   *p = 0 ;
   for (unsigned int i=0; i<seq.size(); i++)
      append_move(p, seq[i]) ;
}
char *moveseq_string(const moveseq *seq) {
   if (*seq.size() > 65)
      error("! can't print a move sequence that long") ;
   char *p = static_buf ;
   append_moveseq(&p, seq) ;
   return static_buf ;
}



char** parseCube(char* p) {

}

