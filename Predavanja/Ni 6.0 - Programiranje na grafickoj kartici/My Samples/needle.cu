/*
Komanda za kompilaciju (prilagoditi verziju arhitekture):
   nvcc -Xcompiler "/openmp" -Xptxas -v -arch=sm_61 -prec-sqrt true -maxrregcount 32 -O2    -o "needle.exe" "./needle.cu"

Komande za pokretanje koda (izabrati jednu od navedenih):
   ./needle.exe 2048 10
   ./needle.exe 6144 10
   ./needle.exe 16384 10
   ./needle.exe 22528 10
   
Napomene:
   + prilagoditi broj streaming multiprocessor-a i core-ova po jednom SM-u (moj GPU ima 28 SM-a i 128 core-ova po SM-u)
   + prilagoditi verziju arhitekture u komandi za kompilaciju (moja arhitektura je sm_61)
   
Objašnjenje komande za kompilaciju (ukloniti komentare iz redova i spojiti ih sve u jedan string):
   nvcc                        // call the nvidia c++ compiler wrapper
      -Xcompiler "/openmp"     // pass the string arguments to the underlying c++ compiler (msvc)
                               // +   use openmp (for timing purposes)
      -Xptxas -v               // show verbose cuda kernel compilation info
      -arch=sm_61              // architecture - cuda 6.1 (replace with your architecture revision number!)
      -prec-sqrt true          // use precise sqrt
      -maxrregcount 32         // maximum registers available per thread
   
   -o "output-file-name.exe"   // output file name, place in quotes ("") if there is a space in the name
   "input-file-name.cu"        // input file name, place in quotes ("") if there is a space in the name
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cooperative_groups.h>
#include "omp.h"
// #define INT_MAX +2147483647
// #define INT_MIN -2147483648
// stream used through the rest of the program
#define STREAM_ID 0
// number of streaming multiprocessors (sm-s) and cores per sm
#define MPROCS 28
#define CORES 128
// number of threads in warp
#define WARPSZ 32
// tile sizes for kernels A and B
// +   tile A should have one dimension be a multiple of the warp size for full memory coallescing
// +   tile B must have one dimension fixed to the number of threads in a warp
const int tileAx = 1*WARPSZ;
const int tileAy = 32;
const int tileBx = 60;
const int tileBy = WARPSZ;

// get the specified element from the given linearized matrix
#define el( mat, cols, i, j ) ( mat[(i)*(cols) + (j)] )


// block substitution matrix
#define BLOSUMSZ 24
int blosum62[BLOSUMSZ][BLOSUMSZ] =
{
   {  4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4 },
   { -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4 },
   { -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4 },
   { -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4 },
   {  0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4 },
   { -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4 },
   { -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4 },
   {  0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4 },
   { -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4 },
   { -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4 },
   { -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4 },
   { -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4 },
   { -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4 },
   { -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4 },
   { -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4 },
   {  1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4 },
   {  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4 },
   { -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4 },
   { -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4 },
   {  0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4 },
   { -2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4 },
   { -1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4 },
   {  0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4 },
   { -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1 }
};

// call in case of invalid command line arguments
void Usage( char* argv[] )
{
   fprintf(stderr, "Usage: %s <rows=cols> <insdelcost>\n", argv[0]);
   fprintf(stderr, "\t<dimension>      - x and y dimensions\n");
   fprintf(stderr, "\t<insdelcost>     - insert and delete cost (positive integer)\n");
   fflush(stderr);
   exit(0);
}


// print one of the optimal matching paths to a file
void Traceback( const char* fname, int* score, int rows, int cols, int adjrows, int adjcols, unsigned* _hash )
{
   printf("   - printing traceback\n");
   
   // if the given file and matrix are null, or the matrix is the wrong size, return
   if( !fname || !score ) return;
   if( rows <= 0 || cols <= 0 ) return;
   // try to open the file with the given name, return if unsuccessful
   FILE *fout = fopen( fname, "w" );
   if( !fout ) return;
   
   // variable used to calculate the hash function
   // http://www.cse.yorku.ca/~oz/hash.html
   // the starting value is a magic constant
   unsigned hash = 5381;

   // for all elements on one of the optimal paths
   bool loop = true;
   for( int i = rows-1, j = cols-1;  loop;  )
   {
      // print the current element
      fprintf( fout, "%d\n", el(score,adjcols, i,j) );
      // add the current element to the hash
      hash = ( ( hash<<5 ) + hash ) ^ el(score,adjcols, i,j);

      int max = INT_MIN;   // maximum value of the up, left and diagonal neighbouring elements
      int dir = '-';       // the current movement direction is unknown

      if( i > 0 && j > 0 && max < el(score,adjcols, i-1,j-1) ) { max = el(score,adjcols, i-1,j-1); dir = 'i'; }   // diagonal movement if possible
      if( i > 0          && max < el(score,adjcols, i-1,j  ) ) { max = el(score,adjcols, i-1,j  ); dir = 'u'; }   // up       movement if possible
      if(          j > 0 && max < el(score,adjcols, i  ,j-1) ) { max = el(score,adjcols, i  ,j-1); dir = 'l'; }   // left     movement if possible

      // move to the neighbour with the maximum value
      switch( dir )
      {
      case 'i': i--; j--; break;
      case 'u': i--;      break;
      case 'l':      j--; break;
      default:  loop = false; break;
      }
   }

   // close the file handle
   fclose( fout );
   // save the hash value
   *_hash = hash;
}



// update the score given the current score matrix and position
void UpdateScore( const int* seqX, const int* seqY, int* score, int rows, int cols, int insdelcost, int i, int j )
{
   int p1 = el(score,cols, i-1,j-1) + blosum62[ seqY[i] ][ seqX[j] ];
   int p2 = el(score,cols, i-1,j  ) - insdelcost;
   int p3 = el(score,cols, i  ,j-1) - insdelcost;
   el(score,cols, i,j) = max( max( p1, p2 ), p3 );
}

// sequential implementation of the Needleman Wunsch algorithm
int Seq( const int* seqX, const int* seqY, int* score, int rows, int cols, int adjrows, int adjcols, int insdelcost, float* time )
{
   // check if the given input is valid, if not return
   if( !seqX || !seqY || !score || !time ) return false;
   if( rows <= 1 || cols <= 1 ) return false;

   // start the timer
   *time = omp_get_wtime();


   // skip the first row and first column in the next calculation
   rows--; cols--;

   // initialize the first row and column of the score matrix
   for( int i = 0; i < 1+rows; i++ ) el(score,adjcols, i,0) = -i*insdelcost;
   for( int j = 0; j < 1+cols; j++ ) el(score,adjcols, 0,j) = -j*insdelcost;

   //  / / / . .
   //  / / . . .
   //  / . . . .
   printf("   - processing top-left triangle + first diagonal of the score matrix\n");
   for( int s = 0; s < rows; s++ )
   for( int t = 0; t <= s; t++ )
   {
      int i = 1 +     t;
      int j = 1 + s - t;
      UpdateScore( seqX, seqY, score, adjrows, adjcols, insdelcost, i, j );
   }

   //  . . . / /
   //  . . / / .
   //  . / / . .
   // if the matrix is not square shaped
   if( rows != cols )
   {
      printf("   - processing all other diagonals of the score matrix\n");
      for( int s = rows; s < cols; s++ )
      for( int t = 0; t <= rows-1; t++ )
      {
         int i = 1 +     t;
         int j = 1 + s - t;
         UpdateScore( seqX, seqY, score, adjrows, adjcols, insdelcost, i, j );
      }
   }

   //  . . . . .|/ /
   //  . . . . /|/
   //  . . . / /|
   printf("   - processing bottom-right triangle of the score matrix\n");
   for( int s = cols; s < cols-1 + rows; s++ )
   for( int t = s-cols+1; t <= rows-1; t++ )
   {
      int i = 1 +     t;
      int j = 1 + s - t;
      UpdateScore( seqX, seqY, score, adjrows, adjcols, insdelcost, i, j );
   }

   // restore the original row and column count
   rows++; cols++;

   // stop the timer
   *time = ( omp_get_wtime() - *time );
   // return that the operation is successful
   return true;
}



// cuda kernel A for the parallel implementation
// +   initializes the score matrix in the gpu
__global__ void kernelA( int* seqX_gpu, int* seqY_gpu, int* score_gpu, int rows, int cols, int (*blosum62_gpu)[BLOSUMSZ], int insdelcost )
{
   // the blosum matrix and relevant parts of the two sequences
   // +   stored in shared memory for faster random access
   __shared__ int blosum62[BLOSUMSZ][BLOSUMSZ];
   __shared__ int seqX[tileAx];
   __shared__ int seqY[tileAy];

   // initialize the blosum shared memory copy
   {
      // map the threads from the thread block onto the blosum matrix elements
      int i = threadIdx.y*BLOSUMSZ + threadIdx.x;
      // while the current thread maps onto an element in the matrix
      while( i < BLOSUMSZ*BLOSUMSZ )
      {
         // copy the current element from the global blosum matrix
         blosum62[ 0 ][ i ] = blosum62_gpu[ 0 ][ i ];
         // map this thread to the next element with stride equal to the number of threads in this block
         i += tileAy*tileAx;
      }
   }

   // initialize the X and Y sequences' shared memory copies
   {
      // position of the current thread in the global X and Y sequences
      int x = blockIdx.x*blockDim.x;
      int y = blockIdx.y*blockDim.y;
      // map the threads from the first            row  to the shared X sequence part
      // map the threads from the second and later rows to the shared Y sequence part
      int iX = ( threadIdx.y     )*tileAx + threadIdx.x;
      int iY = ( threadIdx.y - 1 )*tileAx + threadIdx.x;

      // if the current thread maps to the first row, initialize the corresponding element
      if( iX < tileAx )        seqX[ iX ] = seqX_gpu[ x + iX ];
      // otherwise, remap it to the first column and initialize the corresponding element
      else if( iY < tileAy )   seqY[ iY ] = seqY_gpu[ y + iY ];
   }
   
   // make sure that all threads have finished initializing their corresponding elements
   __syncthreads();

   // initialize the score matrix in global memory
   {
      // position of the current thread in the score matrix
      int i = blockIdx.y*blockDim.y + threadIdx.y;
      int j = blockIdx.x*blockDim.x + threadIdx.x;
      // position of the current thread in the sequences
      int iX = threadIdx.x;
      int iY = threadIdx.y;
      // the current element value
      int elem = 0;
      
      // if the current thread is outside the score matrix, return
      if( i >= rows || j >= cols ) return;

      // if the current thread is not in the first row or column of the score matrix
      // +   use the blosum matrix to calculate the score matrix element value
      // +   increase the value by insert delete cost, since then the formula for calculating the actual element value in kernel B becomes simpler
      if( i > 0 && j > 0 ) { elem = blosum62[ seqY[iY] ][ seqX[iX] ] + insdelcost; }
      // otherwise, if the current thread is in the first row or column
      // +   update the score matrix element using the insert delete cost
      else                 { elem = -( i|j )*insdelcost; }
      
      // update the corresponding element in global memory
      // +   fully coallesced memory access
      el(score_gpu,cols, i,j) = elem;
   }
}

// cuda kernel B for the parallel implementation
// +   calculates the score matrix in the gpu using the initialized score matrix from kernel A
// +   the given matrix minus the padding (zeroth row and column) must be evenly divisible by the tile B
__global__ void kernelB( int* score_gpu, int trows, int tcols, int insdelcost )
{
   // matrix tile which this thread block maps onto
   // +   stored in shared memory for faster random access
   __shared__ int tile[1+tileBy][1+tileBx];

   
   //    |/ / / . .   +   . . . / /   +   . . . . .|/ /
   //   /|/ / . . .   +   . . / / .   +   . . . . /|/
   // / /|/ . . . .   +   . / / . .   +   . . . / /|

   // for all diagonals of tiles in the grid of tiles (score matrix)
   for( int s = 0;   s < tcols-1 + trows;   s++ )
   {
      // (s,t) -- tile coordinates in the grid of tiles (score matrix)
      int tbeg = max( 0, s - (tcols-1) );
      int tend = min( s, trows-1 );


      // map a tile on the current diagonal of tiles to this thread block
      // +   then go to the next tile on the diagonal with stride equal to the number of thread blocks in the thread grid
      for( int t = tbeg + blockIdx.x;   t <= tend;   t += gridDim.x )
      {
         // initialize the score matrix tile
         {
            // position of the top left element of the current tile in the score matrix
            int ibeg = ( 1 + (   t )*tileBy ) - 1;
            int jbeg = ( 1 + ( s-t )*tileBx ) - 1;
            // the number of colums in the score matrix
            int cols = 1 + tcols*tileBx;

            // current thread position in the tile
            int i = threadIdx.x / ( tileBx+1 );
            int j = threadIdx.x % ( tileBx+1 );
            // stride on the current thread position in the tile, equal to the number of threads in this thread block
            // +   it is split into row and column increments for the tread's position for performance reasons (avoids using division and modulo operator in the inner cycle)
            int di = blockDim.x / ( tileBx+1 );
            int dj = blockDim.x % ( tileBx+1 );
            
            // while the current thread maps onto an element in the tile
            while( i < ( 1+tileBy ) )
            {
               // copy the current element from the global score matrix to the tile
               tile[ i ][ j ] = el(score_gpu,cols, ibeg+i,jbeg+j);

               // map the current thread to the next tile element
               i += di; j += dj;
               // if the column index is out of bounds, increase the row index by one and wrap around the column index
               if( j >= ( 1+tileBx ) ) { i++; j -= ( 1+tileBx ); }
            }
         }

         // all threads in this block should finish initializing this tile in shared memory
         __syncthreads();
         
         // calculate the tile elements
         // +   only threads in the first warp from this block are active here, other warps have to wait
         if( threadIdx.x < WARPSZ )
         {
            // the number of rows and colums in the tile without its first row and column (the part of the tile to be calculated)
            int rows = tileBy;
            int cols = tileBx;

            //    |/ / / . .   +   . . . / /   +   . . . . .|/ /
            //   /|/ / . . .   +   . . / / .   +   . . . . /|/
            // / /|/ . . . .   +   . / / . .   +   . . . / /|

            // for all diagonals in the tile without its first row and column
            for( int d = 0;   d < cols-1 + rows;   d++ )
            {
               // (d,p) -- element coordinates in the tile
               int tbeg = max( 0, d - (cols-1) );
               int tend = min( d, rows-1 );
               // position of the current thread's element on the tile diagonal
               int p = tbeg + threadIdx.x;

               // if the thread maps onto an element on the current tile diagonal
               if( p <= tend )
               {
                  // position of the current element
                  int i = 1 + (   p );
                  int j = 1 + ( d-p );
                  
                  // calculate the current element's value
                  // +   always subtract the insert delete cost from the result, since the kernel A added that value to each element of the score matrix
                  int temp1  =      tile[i-1][j-1] + tile[i  ][j  ];
                  int temp2  = max( tile[i-1][j  ] , tile[i  ][j-1] );
                  tile[i][j] = max( temp1, temp2 ) - insdelcost;
               }

               // all threads in this warp should finish calculating the tile's current diagonal
               __syncwarp();
            }
         }
         
         // all threads in this block should finish calculating this tile
         __syncthreads();
         

         // save the score matrix tile
         {
            // position of the first (top left) calculated element of the current tile in the score matrix
            int ibeg = ( 1 + (   t )*tileBy );
            int jbeg = ( 1 + ( s-t )*tileBx );
            // the number of colums in the score matrix
            int cols = 1 + tcols*tileBx;

            // current thread position in the tile
            int i = threadIdx.x / tileBx;
            int j = threadIdx.x % tileBx;
            // stride on the current thread position in the tile, equal to the number of threads in this thread block
            // +   it is split into row and column increments for the tread's position for performance reasons (avoids using division and modulo operator in the inner cycle)
            int di = blockDim.x / tileBx;
            int dj = blockDim.x % tileBx;
            
            // while the current thread maps onto an element in the tile
            while( i < tileBy )
            {
               // copy the current element from the tile to the global score matrix
               el(score_gpu,cols, ibeg+i,jbeg+j) = tile[ 1+i ][ 1+j ];

               // map the current thread to the next tile element
               i += di; j += dj;
               // if the column index is out of bounds, increase the row index by one and wrap around the column index
               if( j >= tileBx ) { i++; j -= tileBx; }
            }
         }
         
         // all threads in this block should finish saving this tile
         // +   block synchronization unnecessary since the tiles on the current diagonal are independent
      }

      // all threads in this grid should finish calculating the diagonal of tiles
      cooperative_groups::this_grid().sync();
   }
}

// parallel implementation of the Needleman Wunsch algorithm
bool Par( const int* seqX, const int* seqY, int* score, int rows, int cols, int adjrows, int adjcols, int insdelcost, float* time, float* ktime )
{
   // check if the given input is valid, if not return
   if( !seqX || !seqY || !score || !time || !ktime ) return false;
   if( rows <= 1 || cols <= 1 ) return false;

   // start the host timer and initialize the gpu timer
   *time = omp_get_wtime();
   *ktime = 0;

   // blosum matrix, sequences which will be compared and the score matrix stored in gpu global memory
   int *blosum62_gpu, *seqX_gpu, *seqY_gpu, *score_gpu;
   // allocate space in the gpu global memory
   cudaMalloc( &seqX_gpu,     adjcols           * sizeof( int ) );
   cudaMalloc( &seqY_gpu,     adjrows           * sizeof( int ) );
   cudaMalloc( &score_gpu,    adjrows*adjcols   * sizeof( int ) );
   cudaMalloc( &blosum62_gpu, BLOSUMSZ*BLOSUMSZ * sizeof( int ) );
   // copy data from host to device
	cudaMemcpy( seqX_gpu,     seqX,     adjcols           * sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( seqY_gpu,     seqY,     adjrows           * sizeof( int ), cudaMemcpyHostToDevice );
	cudaMemcpy( blosum62_gpu, blosum62, BLOSUMSZ*BLOSUMSZ * sizeof( int ), cudaMemcpyHostToDevice );
   // create events for measuring kernel execution time
   cudaEvent_t start, stop;
   cudaEventCreate( &start );
   cudaEventCreate( &stop );


   printf("   - processing score matrix in a blocky diagonal fashion\n");


   // launch kernel A
   {
      // calculate grid dimensions for kernel A
      dim3 gridA;
      gridA.y = ceil( float( adjrows )/tileAy );
      gridA.x = ceil( float( adjcols )/tileAx );
      // block dimensions for kernel A
      dim3 blockA { tileAx, tileAy };
      
      // launch the kernel in the given stream (don't statically allocate shared memory)
      // +   capture events around kernel launch as well
      // +   update the stop event when the kernel finishes
      cudaEventRecord( start, STREAM_ID );
      kernelA<<< gridA, blockA, 0, STREAM_ID >>>( seqX_gpu, seqY_gpu, score_gpu, adjrows, adjcols, ( int (*)[BLOSUMSZ] )blosum62_gpu, insdelcost );
      cudaEventRecord( stop, STREAM_ID );
      cudaEventSynchronize( stop );
      
      // kernel A execution time
      float ktimeA;
      // calculate the time between the given events
      cudaEventElapsedTime( &ktimeA, start, stop );
      // update the total kernel execution time
      *ktime += ktimeA;
   }
   
   // wait for the gpu to finish before going to the next step
   cudaDeviceSynchronize();


   // launch kernel B
   {
      // grid and block dimensions for kernel B
      dim3 gridB;
      dim3 blockB;
      // the number of tiles per row and column of the score matrix
      int trows = ceil( float( adjrows-1 )/tileBy );
      int tcols = ceil( float( adjcols-1 )/tileBx );
      
      // calculate grid and block dimensions for kernel B
      {
         // take the number of warps on the largest tile diagonal times the number of threads in a warp as the number of threads
         // +   also multiply by the number of half warps in the larger dimension for faster writing to global gpu memory
         blockB.x  = ceil( min( tileBy, tileBx )*1./WARPSZ )*WARPSZ;
         blockB.x *= ceil( max( tileBy, tileBx )*2./WARPSZ );
         // take the number of tiles on the largest score matrix diagonal as the only dimension
         gridB.x = min( trows, tcols );

         // the maximum number of parallel blocks on a streaming multiprocessor
         int maxBlocksPerSm = 0;
         // number of threads per block that the kernel will be launched with
         int numThreads = blockB.x;
         // size of shared memory per block in bytes
         int sharedMemSz = ( ( 1+tileBy )*( 1+tileBx ) )*sizeof( int );

         // calculate the max number of parallel blocks per streaming multiprocessor
         cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxBlocksPerSm, kernelB, numThreads, sharedMemSz );
         // the number of cooperative blocks launched must not exceed the maximum possible number of parallel blocks on the device
         gridB.x = min( gridB.x, MPROCS*maxBlocksPerSm );
      }

      // group arguments to be passed to kernel B
      void* kargs[] { &score_gpu, &trows, &tcols, &insdelcost };
      
      // launch the kernel in the given stream (don't statically allocate shared memory)
      // +   capture events around kernel launch as well
      // +   update the stop event when the kernel finishes
      cudaEventRecord( start, STREAM_ID );
      cudaLaunchCooperativeKernel( ( void* )kernelB, gridB, blockB, kargs, 0, STREAM_ID );
      cudaEventRecord( stop, STREAM_ID );
      cudaEventSynchronize( stop );
      
      // kernel B execution time
      float ktimeB;
      // calculate the time between the given events
      cudaEventElapsedTime( &ktimeB, start, stop );
      // update the total kernel execution time
      *ktime += ktimeB;
   }

   // wait for the gpu to finish before going to the next step
   cudaDeviceSynchronize();
   // save the calculated score matrix
   // +   waits for the device to finish, then copies data from device to host
   cudaMemcpy( score, score_gpu, adjrows*adjcols * sizeof( int ), cudaMemcpyDeviceToHost );

   // stop the timer
   *time = ( omp_get_wtime() - *time );

   
   // free allocated space in the gpu global memory
   cudaFree( seqX_gpu );
   cudaFree( seqY_gpu );
   cudaFree( score_gpu );
   cudaFree( blosum62_gpu );
   // free events' memory
   cudaEventDestroy( start );
   cudaEventDestroy( stop );

   // return that the operation is successful
   return true;
}





// main program
int main( int argc, char** argv )
{
   fflush(stdout);
   if( argc != 3 ) Usage( argv );

   // number of rows, number of columns and insdelcost
   int rows = atoi( argv[1] );
   int cols = rows;
   int insdelcost = atoi( argv[2] );
   // add the padding (zeroth row and column) to the matrix
   rows++; cols++;
   // if the number of columns is less than the number of rows, swap them
   if( cols < rows ) { int temp = cols; cols = rows; rows = temp; }

   // adjusted matrix dimensions
   // +   the matrix dimensions are rounded up to 1 + the nearest multiple of the tile B size (in order to be evenly divisible)
   int adjrows = 1 + tileBy*ceil( float( rows-1 )/tileBy );
   int adjcols = 1 + tileBx*ceil( float( cols-1 )/tileBx );

   // allocate memory for the sequences which will be compared and the score matrix
   int* const seqX  = ( int* ) malloc( adjcols * sizeof( int ) );
   int* const seqY  = ( int* ) malloc( adjrows * sizeof( int ) );
   int* const score = ( int* ) malloc( adjrows*adjcols * sizeof( int ) );

   // if memory hasn't been allocated
   if( !seqX || !seqY || !score )
   {
      fprintf(stderr, "Error: memory allocation failed\n");
      fflush(stderr);

      // free allocated memory
      free( seqX ); free( seqY ); free( score );
      exit(1);
   }

   // seed the random generator
// unsigned int seed = time( NULL );
   unsigned int seed = 1605868371;
   srand( seed );

   // initialize the sequences A and B to random values in the range [1-10]
   // +   also initialize the padding with zeroes
   seqX[0] = 0;
   seqY[0] = 0;
   for( int j = 1; j < adjcols; j++ ) seqX[j] = ( j < cols ) ? 1 + rand() % 10 : 0;
   for( int i = 1; i < adjrows; i++ ) seqY[i] = ( i < rows ) ? 1 + rand() % 10 : 0;

   // variables for measuring the algorithms' cpu execution time and kernel execution time
   float htime = 0, ktime = 0;
   // variables for storing the calculation hashes
   unsigned hash1 = 10, hash2 = 20;

   // use the Needleman-Wunsch algorithm to find the optimal matching between the input vectors
   // +   sequential implementation
   printf("Sequential implementation:\n" );
   Seq( seqX, seqY, score, rows, cols, adjrows, adjcols, insdelcost, &htime );
   Traceback( "needle.out1.txt", score, rows, cols, adjrows, adjcols, &hash1 );
   printf("   hash=%10u\n", hash1 );
   printf("   time=%9.6fs\n", htime );
   fflush(stdout);
   
   // +   parallel implementation
   printf("Parallel implementation:\n" );
   Par( seqX, seqY, score, rows, cols, adjrows, adjcols, insdelcost, &htime, &ktime );
   Traceback( "needle.out2.txt", score, rows, cols, adjrows, adjcols, &hash2 );
   printf("   hash=%10u\n", hash2 );
   printf("   time=%9.6fs ktime=%9.6fs\n", htime, ktime );
   fflush(stdout);

   // +   compare the implementations
   if( hash1 == hash2 ) printf( "TEST PASSED\n" );
   else                 printf( "TEST FAILED\n" );
   fflush(stdout);

   // free allocated memory
   free( seqX ); free( seqY );
}


