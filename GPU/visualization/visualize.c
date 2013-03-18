// Include files

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_math.h>

// Constant definitions.

//now taken from main.cu
//#define GRAPH_SIZE 5

#define DRAWGRAPH_GRIDSIZE 1

const int  POINTS_PER_GRAPH_NODE= 4;
const int  POINTS_PER_GRAPH_LINE= 2;
const int  POINTS_PER_ANT= 3;


//PROPERTIES FOR THE GRAPH NODES
//the number of levels in the x direction
#define NLEVELS 8
//the number of nodes at each level
const int levels[NLEVELS] = {4,4,4,4,4,4,4,4};
//the size (width) of one node square
const float NODE_SIZE = 0.02;
//the distance between each node in the Y direction
#define NODE_Y_DIST 0.2
#define NODE_X_DIST 0.2



//PROPERTIES ABOUT THE ANTS

//now taken from main.cu
//#define NB_ANT 3

//now defined in main.cu
// int h_solutions[NB_ANT * GRAPH_SIZE] = {
//                                         0,2,3,4,0,
//                                         0,1,3,0,0,
//                                         0,3,4,0,0
//                                         };
// int *d_solutions;


//NOTHING TO CHANGE HERE
const int GRAPH_ELEMENT_BUFFER_SIZE = GRAPH_SIZE * POINTS_PER_GRAPH_NODE /*FOR THE GRAPH NODES*/ 
                                 + GRAPH_SIZE *  GRAPH_SIZE  * POINTS_PER_GRAPH_LINE;  /*FOR THE GRAPH LINES*/

const int GRAPH_ARRAY_BUFFER_SIZE = GRAPH_SIZE * POINTS_PER_GRAPH_NODE * 2 /*FOR THE GRAPH NODES*/ 
                                 + GRAPH_SIZE *  GRAPH_SIZE  * POINTS_PER_GRAPH_LINE * 2;  /*FOR THE GRAPH LINES*/

const int ANT_ELEMENT_BUFFER_SIZE = NB_ANT * POINTS_PER_ANT ;

const int ANT_ARRAY_BUFFER_SIZE = NB_ANT * POINTS_PER_ANT * 2;

const int  TOTAL_ELEMENT_ARRAY_BUFFER_SIZE = GRAPH_ELEMENT_BUFFER_SIZE
                                 + ANT_ELEMENT_BUFFER_SIZE;                          /* For the ants */

const int  TOTAL_ARRAY_BUFFER_SIZE = GRAPH_ARRAY_BUFFER_SIZE
                                 + ANT_ARRAY_BUFFER_SIZE ;                          /* For the ants */



// Declarations.
#define glSafeCall(x) do {x; int _i = glGetError();     \
    if(_i != GL_NO_ERROR) gl_error_abort(_i, __LINE__, __FILE__); } while(0)
void gl_error_abort(int, int, char*);
void display(void);
int animate(void);
void redisplay(int);
void cleanup(void);
__global__ void updateGraphNodevbo(float2*, float*, float*);
__global__ void updateGraphLinevbo(float2*, int, float*, float*, int*) ;
__global__ void updateAntvbo(float2* d_vbo , int offset, float* d_ant_x,float* d_ant_y,float* d_ant_dx,float* d_ant_dy);

void init_graph(int * G, int size);


float *h_graphx, *h_graphy, *d_graphx, *d_graphy;

//for the ants
float *h_ant_x, *d_ant_x, *h_ant_y, *d_ant_y, *h_ant_dx, *d_ant_dx, *h_ant_dy, *d_ant_dy;
int *h_current_ant_destination;
int *d_current_ant_destination;


// Don't worry about globals below here, they're for rendering.
float2 *d_vbo;
float2 h_vbo[TOTAL_ELEMENT_ARRAY_BUFFER_SIZE];

GLuint h_vbo_handle, h_ibo_handle;

// short h_ibo[TOTAL_ELEMENT_ARRAY_BUFFER_SIZE];
struct cudaGraphicsResource *bridge; // connection between openGL and CUDA

//holding the size of buffer memory required

//The graph distance matrix
int * d_graph_matrix, *h_graph_matrix ;



__global__ void update_ant_step(int* d_current_ant_destination, float* d_graphx,float* d_graphy,float *x, float *y, float *dx, float *dy, int *d_solutions)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

    int  index, gindex, pindex ;
    
      float xt = x[tid];
      float yt = y[tid];
       

      index = tid * GRAPH_SIZE + d_current_ant_destination[tid];
      gindex = d_solutions[index];
      pindex = d_solutions[index-1];
      

      if ( gindex != 0) {

       


        dx[tid] = (  (d_graphx[ gindex ]) - d_graphx[ pindex ] + NODE_SIZE)/300 ;
        dy[tid] = ( ( d_graphy[ gindex ]) - d_graphy[ pindex ] )/300 ;

        

      xt += dx[tid];
      yt += dy[tid];
      if (xt > 1.0) xt -= 2.0;
      if (yt > 1.0) yt -= 2.0;
      if (xt < -1.0) xt += 2.0;
      if (yt < -1.0) yt += 2.0;

        if (xt >= d_graphx[ gindex ]- NODE_SIZE ) {
          d_current_ant_destination[tid] += 1;
          xt -= NODE_SIZE;
       }

      x[tid] = xt;
      y[tid] = yt;

    }
}


__global__ void drawNodes(float *graphx, float *graphy, int nodeOffset, float startX, float startY, float dstX, float dstY )
{
    int tid = threadIdx.x ;
    graphx[  tid + nodeOffset ] = startX + dstX * (tid );
    graphy[  tid + nodeOffset ] = startY + dstY * (tid );
}




// Initialization.
int init_visualization(int argc,char ** argv)
{

printf("%d\n================================\n", TOTAL_ARRAY_BUFFER_SIZE * sizeof(float));
  //int device_id;

  // initialise OpenGL, don't worry about this
printf("asfasfdsdafasdfsdf\n");
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(1024,1024);
  glutCreateWindow("ACO"); // feel free to change this
  glewInit();
  // initialise event handlers
   glutDisplayFunc(display);
  atexit(cleanup);


  // // intialise CUDA
  //device_id = cutGetMaxGflopsDeviceId();
  //cutilSafeCall(cudaGLSetGLDevice(device_id));

  // allocate memory for x, y, dx, dy
  // you can add more state arrays if you like
  h_ant_x = (float *)malloc(NB_ANT*sizeof(float));
  cutilSafeCall(cudaMalloc((void **)&d_ant_x, NB_ANT*sizeof(float)));
  h_ant_y = (float *)malloc(NB_ANT*sizeof(float));
  cutilSafeCall(cudaMalloc((void **)&d_ant_y, NB_ANT*sizeof(float)));
  h_ant_dx = (float *)malloc(NB_ANT*sizeof(float));
  cutilSafeCall(cudaMalloc((void **)&d_ant_dx, NB_ANT*sizeof(float)));
  h_ant_dy = (float *)malloc(NB_ANT*sizeof(float));
  cutilSafeCall(cudaMalloc((void **)&d_ant_dy, NB_ANT*sizeof(float)));




  h_graphx = (float *)malloc(GRAPH_SIZE  * sizeof(float));
  cutilSafeCall(cudaMalloc((void **)&d_graphx, GRAPH_SIZE   * sizeof(float)));
  h_graphy = (float *)malloc(GRAPH_SIZE  * sizeof(float));
  cutilSafeCall(cudaMalloc((void **)&d_graphy, GRAPH_SIZE * sizeof(float)));


  //allocate d_solutions
  d_solutions = (int*)malloc(sizeof(int) * NB_ANT * GRAPH_SIZE);
  cutilSafeCall(cudaMalloc((void **)&d_solutions, sizeof(int) * NB_ANT * GRAPH_SIZE));
  cutilSafeCall(cudaMemcpy(d_solutions, h_solutions, sizeof(int) * NB_ANT * GRAPH_SIZE, cudaMemcpyHostToDevice));




  //the current node destination
  h_current_ant_destination = (int*)malloc(sizeof(int) * NB_ANT);
  for (short i=0; i<NB_ANT; i++) {
    h_current_ant_destination[i] = 1;
  }
  //allocate current node destinations
    cutilSafeCall(cudaMalloc((void **)&d_current_ant_destination, sizeof(int) * NB_ANT));
  cutilSafeCall(cudaMemcpy(d_current_ant_destination, h_current_ant_destination, sizeof(int) * NB_ANT , cudaMemcpyHostToDevice));

   h_graph_matrix = (int *)malloc( sizeof(int)*GRAPH_SIZE * GRAPH_SIZE);
  cutilSafeCall(cudaMalloc((void **)&d_graph_matrix, GRAPH_SIZE * GRAPH_SIZE * sizeof(int)));
  init_graph(h_graph_matrix, GRAPH_SIZE);
  cutilSafeCall(cudaMemcpy(d_graph_matrix, h_graph_matrix, GRAPH_SIZE * GRAPH_SIZE * sizeof(int), cudaMemcpyHostToDevice));




  for (short i = 0; i < TOTAL_ELEMENT_ARRAY_BUFFER_SIZE; i++) {
    h_vbo[i].x = 0;
    h_vbo[i].y = 0;
  }

  glSafeCall( glGenBuffers(1, &h_vbo_handle) );
  // glSafeCall( glGenBuffers(1, &h_ibo_handle) );
  glSafeCall( glBindBuffer(GL_ARRAY_BUFFER, h_vbo_handle) );
  // glSafeCall( glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, h_ibo_handle) );
  glSafeCall( glBufferData(GL_ARRAY_BUFFER,  TOTAL_ARRAY_BUFFER_SIZE * sizeof(float), h_vbo,GL_DYNAMIC_COPY) );
  // glSafeCall( glBufferData(GL_ELEMENT_ARRAY_BUFFER,  TOTAL_ELEMENT_ARRAY_BUFFER_SIZE* sizeof(short),
  //        h_ibo, GL_STATIC_DRAW) );


  // run main loop
  glutMainLoop();
  return 0;
}

// Main loop
void redisplay(int unused) { (void)unused; animate(); }
void display(void) {
  size_t unused;
int i;



  // execute your kernel
  int offset = 0;

  for (i=0;i<NLEVELS;i++) { 
  printf("%d\n", offset);
  drawNodes<<< DRAWGRAPH_GRIDSIZE, levels[i]>>> 
  (d_graphx, d_graphy, offset, -1 + i * NODE_X_DIST , 0 - levels[i] * NODE_Y_DIST/2 , 0, NODE_Y_DIST);
  offset += levels[i];

  cutilCheckMsg("updatestep execution failed");
  }


cutilSafeCall(cudaMemcpy(h_graphx, d_graphx, GRAPH_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

cutilSafeCall(cudaMemcpy(h_graphy, d_graphy, GRAPH_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

  // initizlize all the ants to start at the first graph node
  for (i = 0; i < NB_ANT; i++) {
    int firstnode = h_solutions[i * GRAPH_SIZE + 0];
    srand(time(NULL));
    h_ant_x[i] =  h_graphx[firstnode] ;//h_graphx[firstnode];
    h_ant_y[i] =  h_graphy[firstnode] ;//h_graphy[firstnode];
    h_ant_dx[i] = 0.003;
    h_ant_dy[i] = 0.003;
  }
 

cutilSafeCall(cudaMemcpy( d_ant_x, h_ant_x, NB_ANT * sizeof(float), cudaMemcpyHostToDevice));

cutilSafeCall(cudaMemcpy( d_ant_y, h_ant_y, NB_ANT * sizeof(float), cudaMemcpyHostToDevice));


  cutilSafeCall( cudaMemcpy(d_ant_x,h_ant_x,NB_ANT*sizeof(float),
                 cudaMemcpyHostToDevice) );
  cutilSafeCall( cudaMemcpy(d_ant_y,h_ant_y,NB_ANT*sizeof(float),
                 cudaMemcpyHostToDevice) );
  cutilSafeCall( cudaMemcpy(d_ant_dx,h_ant_dx,NB_ANT*sizeof(float),
                 cudaMemcpyHostToDevice) );
  cutilSafeCall( cudaMemcpy(d_ant_dy,h_ant_dy,NB_ANT*sizeof(float),
                 cudaMemcpyHostToDevice) );



printf("============\n");

for (i=0;i<GRAPH_SIZE;i++) {
    printf("%f , %f ", h_graphx[i], h_graphy[i]);

  printf("\n");
}
printf("---------------\n");





  // execute the CUDA part of redraw code, don't worry about it
  cutilSafeCall( cudaGraphicsGLRegisterBuffer(
                 &bridge, h_vbo_handle, cudaGraphicsMapFlagsNone) );
  cutilSafeCall( cudaGraphicsMapResources(1, &bridge, 0) );
  cutilSafeCall( cudaGraphicsResourceGetMappedPointer(
    (void**)&d_vbo, &unused, bridge) );



  //NOW UPDATE THE VBO ARRAY WITH GRAPH NODES AND LINES
  updateGraphNodevbo <<< DRAWGRAPH_GRIDSIZE, GRAPH_SIZE >>>(d_vbo, d_graphx, d_graphy);
  cutilCheckMsg("updateGraphNodevbo execution failed");


  updateGraphLinevbo <<< DRAWGRAPH_GRIDSIZE, GRAPH_SIZE >>>( d_vbo ,GRAPH_SIZE * POINTS_PER_GRAPH_NODE, d_graphx, d_graphy, d_graph_matrix);
  cutilCheckMsg("updateGraphLinevbo execution failed");

  //AND ANTS
  updateAntvbo <<< DRAWGRAPH_GRIDSIZE, NB_ANT >>>( d_vbo , GRAPH_ELEMENT_BUFFER_SIZE , d_ant_x, d_ant_y, d_ant_dx, d_ant_dy);
  cutilCheckMsg("updateGraphLinevbo execution failed");





  //print the vbo array

  cutilSafeCall(cudaMemcpy(
    h_vbo, d_vbo, sizeof(float) * TOTAL_ARRAY_BUFFER_SIZE , cudaMemcpyDeviceToHost));



  // h_vbo[21].y =1;
  for (i = 0;i<GRAPH_SIZE * POINTS_PER_GRAPH_NODE; i++) {
      printf("%f, %f \n   ", h_vbo[i].x, h_vbo[i].y  );
  }
    printf("\n==========\n" );

  for (i = GRAPH_SIZE * POINTS_PER_GRAPH_NODE;i< GRAPH_SIZE * POINTS_PER_GRAPH_NODE + GRAPH_SIZE * GRAPH_SIZE * POINTS_PER_GRAPH_LINE; i++) {
      printf("%f, %f \n ", h_vbo[i].x, h_vbo[i].y  );
  }
    printf("\n============\n" );




  cutilSafeCall( cudaGraphicsUnmapResources(1, &bridge, 0) );
  cutilSafeCall( cudaGraphicsUnregisterResource(bridge) );

  // execute the OpenGL part of redraw code, don't worry about it either
  glSafeCall( glBindBuffer(GL_ARRAY_BUFFER, h_vbo_handle) );
  glSafeCall( glClear(GL_COLOR_BUFFER_BIT) );
  glSafeCall( glDisable(GL_DEPTH_TEST) );
  glSafeCall( glDisable(GL_CULL_FACE) );
  glSafeCall( glEnable(GL_SCISSOR_TEST) );
  glSafeCall( glEnableClientState(GL_VERTEX_ARRAY) );   // use VBO
  glSafeCall( glVertexPointer(2, GL_FLOAT, 0, 0) );     // specify format of VBO


 
 glSafeCall( glDrawArrays(GL_QUADS, 0, GRAPH_SIZE  * POINTS_PER_GRAPH_NODE) );// actual drawing


 //draw the lines with an offset from the base array
 glSafeCall( glDrawArrays(GL_LINES,   GRAPH_SIZE  * POINTS_PER_GRAPH_NODE  , GRAPH_SIZE * GRAPH_SIZE  * POINTS_PER_GRAPH_LINE ) );// actual drawing
 //draw the ants
 glSafeCall( glDrawArrays(GL_TRIANGLES,  GRAPH_ELEMENT_BUFFER_SIZE , ANT_ELEMENT_BUFFER_SIZE ) );// actual drawing



  glSafeCall( glDisableClientState(GL_VERTEX_ARRAY) );
  glSafeCall( glBindBuffer(GL_ARRAY_BUFFER, 0) );
  // glSafeCall( glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0) );
  glSafeCall( glutSwapBuffers() );
  

  
  animate();
}

int animate(void) {

  size_t unused;

  // execute your kernel
  update_ant_step<<<DRAWGRAPH_GRIDSIZE, NB_ANT>>>
   (d_current_ant_destination,d_graphx, d_graphy, d_ant_x, d_ant_y, d_ant_dx, d_ant_dy, d_solutions);
  cutilCheckMsg("updatestep execution failed");




  // execute the CUDA part of redraw code, don't worry about it
  cutilSafeCall( cudaGraphicsGLRegisterBuffer(
                 &bridge, h_vbo_handle, cudaGraphicsMapFlagsNone) );
  cutilSafeCall( cudaGraphicsMapResources(1, &bridge, 0) );
  cutilSafeCall( cudaGraphicsResourceGetMappedPointer(
    (void**)&d_vbo, &unused, bridge) );
  
  updateAntvbo <<< DRAWGRAPH_GRIDSIZE, NB_ANT >>>( d_vbo , GRAPH_ELEMENT_BUFFER_SIZE , d_ant_x, d_ant_y, d_ant_dx, d_ant_dy);
  cutilCheckMsg("updateGraphLinevbo execution failed");


  cutilCheckMsg("updateGraphNodevbo execution failed");
  cutilSafeCall( cudaGraphicsUnmapResources(1, &bridge, 0) );
  cutilSafeCall( cudaGraphicsUnregisterResource(bridge) );

  // execute the OpenGL part of redraw code, don't worry about it either
  glSafeCall( glBindBuffer(GL_ARRAY_BUFFER, h_vbo_handle) );
  glSafeCall( glClear(GL_COLOR_BUFFER_BIT) );
  glSafeCall( glDisable(GL_DEPTH_TEST) );
  glSafeCall( glDisable(GL_CULL_FACE) );
  glSafeCall( glEnable(GL_SCISSOR_TEST) );
  glSafeCall( glEnableClientState(GL_VERTEX_ARRAY) );   // use VBO
  glSafeCall( glVertexPointer(2, GL_FLOAT, 0, 0) );     // specify format of VBO


 
 glSafeCall( glDrawArrays(GL_QUADS, 0, GRAPH_SIZE  * POINTS_PER_GRAPH_NODE) );// actual drawing


 //draw the lines with an offset from the base array
 glSafeCall( glDrawArrays(GL_LINES,   GRAPH_SIZE  * POINTS_PER_GRAPH_NODE  , GRAPH_SIZE * GRAPH_SIZE  * POINTS_PER_GRAPH_LINE ) );// actual drawing
 //draw the ants

 glSafeCall( glDrawArrays(GL_TRIANGLES,  GRAPH_ELEMENT_BUFFER_SIZE , ANT_ELEMENT_BUFFER_SIZE ) );// actual drawing



  glSafeCall( glDisableClientState(GL_VERTEX_ARRAY) );
  glSafeCall( glBindBuffer(GL_ARRAY_BUFFER, 0) );
  glSafeCall( glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0) );
  glSafeCall( glutSwapBuffers() );
  


  // move to next step of simulation
  

  //check that all the iterations were not yet reached before redoing the animation again

  cutilSafeCall(cudaMemcpy(h_current_ant_destination, d_current_ant_destination, sizeof(int) * NB_ANT, cudaMemcpyDeviceToHost));
  bool allAntsDone = true;
  for (short i=0;i<NB_ANT;i++)
  {
    printf("%d\n", h_solutions[ i * GRAPH_SIZE + h_current_ant_destination[i] ]);
    if (h_solutions[ i * GRAPH_SIZE + h_current_ant_destination[i] ] != 0)
      allAntsDone = false;
  }

  if (!allAntsDone) {   
    glutTimerFunc(16, redisplay, 0); //16ms between frames; you may change this
  }
  else {
    printf("DONE!!!!!!!!!!!!!!!!!!!!!\n");
    return 0;
  }


}



__global__ void updateGraphNodevbo(float2 *vbodata, float *x, float *y)
{
  int tid =  threadIdx.x;
    vbodata[ tid * 4 ].x    =    x[tid];
    vbodata[ tid * 4 ].y    =    y[tid];

    vbodata[ tid * 4 + 1].x =   x[tid] + NODE_SIZE;
    vbodata[ tid * 4 + 1].y =   y[tid];

    vbodata[ tid * 4 + 2].x =   x[tid] + NODE_SIZE;
    vbodata[ tid * 4 + 2].y =   y[tid] + NODE_SIZE;

    vbodata[ tid * 4 + 3].x =   x[tid];
    vbodata[ tid * 4 + 3].y =   y[tid] + NODE_SIZE;


}

__global__ void updateGraphLinevbo(float2 *vbodata, int offset, float *x, float *y, int * graph)
{
  int tid = threadIdx.x;
  int i=0, index ;


    for (i=0; i<GRAPH_SIZE; i++) {
      if ( graph[i * GRAPH_SIZE + tid]  == 1) {
        
        index = offset +i * GRAPH_SIZE * 2 + tid * 2;
        
        vbodata[index].x = x[tid] + NODE_SIZE/2 ;
        vbodata[index].y = y[tid] + NODE_SIZE/2 ;

        vbodata[index + 1 ].x = x[i] + NODE_SIZE/2 ;
        vbodata[index + 1 ].y = y[i] + NODE_SIZE/2 ;
      } 
      __syncthreads();
    }
}

__global__ void updateAntvbo(float2* vbodata , int offset, float* d_ant_x,float* d_ant_y,float* d_ant_dx,float* d_ant_dy) {


  int tid =  threadIdx.x;
  float2 d;
  float xt = d_ant_x[tid];
  float yt = d_ant_y[tid];
  d.x = d_ant_dx[tid];
  d.y = d_ant_dy[tid];
  d = normalize(d);
  d.x *= 0.06;
  d.y *= 0.06;
    vbodata[ tid * 3  + offset].x    =   xt;
    vbodata[ tid * 3  + offset].y    =   yt;

    vbodata[ tid * 3  + offset+ 1].x =   xt - d.x + (d.y * 0.3);
    vbodata[ tid * 3  + offset+ 1].y =   yt - d.y - (d.x * 0.3);

    vbodata[ tid * 3  + offset+ 2].x =  xt - d.x - (d.y * 0.3);
    vbodata[ tid * 3  + offset+ 2].y =  yt - d.y + (d.x * 0.3);



  //   __shared__ float2 vbd[UPDATEVBO_BLOCKSIZE*3];
  // int tid = threadIdx.x + UPDATEVBO_BLOCKSIZE*blockIdx.x;
  // int otid = threadIdx.x + UPDATEVBO_BLOCKSIZE*blockIdx.x*3;
  // int tid3 = threadIdx.x*3;
  // float xt = x[tid];
  // float yt = y[tid];
  // float2 d;
  // d.x = dx[tid];
  // d.y = dy[tid];
  // vbd[tid3].x = xt;
  // vbd[tid3].y = yt;
  // d = normalize(d);
  // d.x *= 0.02;
  // d.y *= 0.02;
  // vbd[tid3+1].x = xt - d.x + (d.y * 0.3);
  // vbd[tid3+1].y = yt - d.y - (d.x * 0.3);
  // vbd[tid3+2].x = xt - d.x - (d.y * 0.3);
  // vbd[tid3+2].y = yt - d.y + (d.x * 0.3);
  // __syncthreads();
  // vbodata[otid] = vbd[threadIdx.x];
  // vbodata[otid+UPDATEVBO_BLOCKSIZE] = vbd[threadIdx.x+UPDATEVBO_BLOCKSIZE];
  // vbodata[otid+UPDATEVBO_BLOCKSIZE*2] = vbd[threadIdx.x+UPDATEVBO_BLOCKSIZE*2];
}



// Error handling for OpenGL.
void gl_error_abort(int i, int line, char *file) {
  char *reason = "unknown reason";
  switch(i) {
  case GL_INVALID_ENUM: reason = "invalid enum"; break;
  case GL_INVALID_VALUE: reason = "invalid value"; break;
  case GL_INVALID_OPERATION: reason = "invalid operation"; break;
  case GL_STACK_OVERFLOW: reason = "stack overflow"; break;
  case GL_STACK_UNDERFLOW: reason = "stack underflow"; break;
  case GL_OUT_OF_MEMORY: reason = "out of memory"; break;
  case GL_TABLE_TOO_LARGE: reason = "table too large"; break;
  }
  fprintf(stderr, "%s(%d) : glSafeCall() runtime error: %s\n",
    file, line, reason);
  exit(1);
}

// Deinitialization.
void cleanup(void) {
  // free memory
  cudaFree(d_graph_matrix);
  free(h_graph_matrix);
  cudaFree(d_graphx);
  free(h_graphx);
  cudaFree(d_graphy);
  free(h_graphy);

  // free OpenGL resources
  glSafeCall( glBindBuffer(GL_ARRAY_BUFFER, 0) );
  glSafeCall( glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0) );
  glSafeCall( glDeleteBuffers(1, &h_vbo_handle) );
  glSafeCall( glDeleteBuffers(1, &h_ibo_handle) );

  cutilDeviceReset();
}




void init_graph(int * G, int size)
{
    int i,j,index;
    for(i=0 ; i<size ; i++)
    {
        for(j=0 ; j<size ; j++)
        {
            index = size*i + j;
            if(i < j)
                G[index] = 1;
            else{
            G[index] = 0;}
            printf ("%d", G[index]);
        }
        printf("\n");
    }
}


