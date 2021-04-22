/*
Komanda za kompilaciju (prilagoditi verziju arhitekture):
   nvcc -Xcompiler "/openmp" -Xptxas -v -arch=sm_61 -prec-sqrt true -maxrregcount 32 -O2    -o "pi.exe" "./pi.cu"

Komande za pokretanje koda (izabrati jednu od navedenih):
   ./pi.exe 1000000
   ./pi.exe 10000000
   ./pi.exe 100000000
   ./pi.exe 1000000000
   ./pi.exe 10000000000
   
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
#include <math.h>
#include "omp.h"
// stream used through the rest of the program
#define STREAM_ID 0
// number of streaming multiprocessors (sm-s) and cores per sm
#define MPROCS 28
#define CORES 128
// number of threads in warp
#define WARPSZ 32

// pozvati u slucaju nevalidnih argumenata na komandnoj liniji
void Usage(char* prog_name)
{
   fprintf(stderr, "usage: %s <thread_count> <n>\n", prog_name);
   fprintf(stderr, "   n is the number of terms and should be >= 1\n");
   fflush(stderr);
   exit(0);
}

// zlatni standard
double Gold( float* time )
{
   // start the timer
   *time = omp_get_wtime();

   double pi = 4.0 * atan(1.0);

   // stop the timer
   *time = ( omp_get_wtime() - *time );
   // return the reference approximation of pi
   return pi;
}

// sekvencijalna implementacija
double Seq( long long n, float* time )
{
   // start the timer
   *time = omp_get_wtime();
   
   double factor = 0.0;
   double sum = 0.0;

   for( long long i = 0; i < n; i++)
   {
      factor = (i % 2 == 0) ? 1.0 : -1.0; 
      sum += factor/(2*i+1);
   }

   // stop the timer
   *time = ( omp_get_wtime() - *time );
   // return the approximation of pi
   return 4.0*sum;
}

// kernel paralelne implementacije
__global__ void kernel( long long n, int iters, double* res )
{
   // partial sums calculated by the threads in this thread's block
   extern __shared__ double sum[];

   // thread iteration boundaries
   long long ibeg = ( blockIdx.x*blockDim.x + threadIdx.x )*( long long )iters;
   long long iend = min( ibeg + iters-1, n-1 );

   // partial sum and current element sign (positive for elements in even positions)
   // +   speedup 10x if these variables are float (probably not losing too much precision anyway)
   float psum = 0;
   float sign = 1 - ( iend&1 << 1 );
   
   // calculate the partial sum for this thread in the thread block
   // +   calculated in reverse in order to minimize floating point rounding errors
   for( ; iend >= ibeg; iend-- )
   {
      psum += sign/( ( iend<<1 ) + 1 );
      sign = -sign;
   }

   // save the partial sum to the thread block shared memory
   sum[threadIdx.x] = psum;

   // reduce the partial sums into a single total sum for this block
   // +   naive reduction on purpose, since the results' precision heavily depends on summing the elements in order
   for( int stride = 1; stride < blockDim.x; stride <<= 1 )
   {
      // synchronize the threads in the block
      __syncthreads();
      // each active thread sums two stride-adjacent elements per iteration
      // +   tid%( stride<<1 ) == 0
      if( !( threadIdx.x&( ( stride<<1 )-1 ) ) )sum[threadIdx.x] += sum[threadIdx.x + stride];
   }

   // save the block partial sum to the result array
   if( threadIdx.x == 0 ) res[blockIdx.x] = sum[threadIdx.x];
}

// paralelna implementacija
double Par( long long n, float* time, float* ktime )
{
   // start the host timer
   *time = omp_get_wtime();

   // grid dimensions and number of elements to be summed per thread before reduction
   // +   these are optimal parameters for minimizing execution overhead, but not for the final result precision
	dim3 gridDim( 4*MPROCS, 1 );
   dim3 blockDim( 4*CORES );
   int iters = 4 * ceil( float( n ) /gridDim.x/blockDim.x/4 );
   // if there are too many iterations per core, we lose precision
   // +   first increase block size, then increase block count (since the cpu must sum the block partial sums)
   iters = min( iters, 20 );
   gridDim.x = ceil( float( n ) /blockDim.x/iters );

   // size of shared memory per block
   int sharedsz = blockDim.x*sizeof( double );
   
   // unified memory for holding the gpu block partial sums
   double* res;
   // allocate unified memory
   cudaMallocManaged( &res, gridDim.x*sizeof( double ) );
   

   // create events for measuring kernel execution time
   cudaEvent_t start, stop;
   // capture event before kernel launch
   cudaEventCreate( &start );
   cudaEventRecord( start, STREAM_ID );
   
   // launch the kernel in the given stream
   kernel<<< gridDim, blockDim, sharedsz, STREAM_ID >>>( n, iters, res );
   // capture kernel launch event
   cudaEventCreate( &stop );
   cudaEventRecord( stop, STREAM_ID );

   // update the stop event when the kernel finishes
   cudaEventSynchronize( stop );
   // calculate the time between the given events
   cudaEventElapsedTime( ktime, start, stop );
   
   
   // wait for the gpu to finish before accessing unified memory on the host
   cudaDeviceSynchronize();
   
   // sum of the partial sums returned from the gpu
   double sum = 0;
   // calculated in reverse in order to minimize floating point rounding errors (adding numbers with very different exponents)
   for( int i = gridDim.x-1; i >= 0 ; i-- ) sum += res[i];
   
   // free unified memory
   cudaFree( res );
   // free events' memory
   cudaEventDestroy( start );
   cudaEventDestroy( stop );

   
   // stop the host timer
   *time = ( omp_get_wtime() - *time );
   // return the approximation of pi
   return 4.0*sum;
}


// glavni program
int main(int argc, char* argv[])
{
   fflush(stdout);
   long long n = 0;

   // provera argumenata
   if( argc != 2 ) Usage( argv[0] );
   n = strtoll( argv[1], NULL, 10 );
   if( n < 1 ) Usage( argv[0] );

	// dummy call to initialize the cuda environment
   cudaDeviceSynchronize();

   printf("With n = %lld terms\n", n);
   double pi1, pi2, pi3;
   float time, ktime;

   pi1 = Gold( &time );
   printf("   reference  pi=%.14f time=%9.6fs\n", pi1, time );
   fflush(stdout);

   pi2 = Seq( n, &time );
   printf("   sequential pi=%.14f time=%9.6fs\n", pi2, time );
   fflush(stdout);

   pi3 = Par( n, &time, &ktime );
   printf("   parallel   pi=%.14f time=%9.6fs  ktime=%9.6fs\n", pi3, time, ktime );
   fflush(stdout);

   if( fabs( pi3 - pi2 ) < 0.000001 ) printf( "TEST PASSED\n" );
   else                               printf( "TEST FAILED\n" );
   fflush(stdout);

   return 0;
}

