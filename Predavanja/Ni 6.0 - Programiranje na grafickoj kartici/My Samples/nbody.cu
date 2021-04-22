/*
Komanda za kompilaciju (prilagoditi verziju arhitekture):
   nvcc -Xcompiler "/openmp" -Xptxas -v -arch=sm_61 -prec-sqrt true -maxrregcount 64 -O2    -o "nbody.exe" "./nbody.cu"

Komande za pokretanje koda (izabrati jednu od navedenih):
   ./nbody.exe 100 500 0.01 500 g
   ./nbody.exe 500 500 0.01 500 g
   ./nbody.exe 1000 500 0.01 500 g
   ./nbody.exe 5000 500 0.01 500 g
   
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
// block and tile sizes for the kernel A
const int tilesz = 8*WARPSZ;
// two-dimensional coordinate system
#define DIM 2
#define X 0
#define Y 1


// 2D vector
typedef double Vect2D[ DIM ];

// structure that describes a particle
typedef struct
{
   double m;    // mass of particle
   Vect2D s;    // position of particle
   Vect2D v;    // velocity of particle
} Particle;

// gravitational constant
#define G (6.673e-11)



// call in case of invalid command line arguments
void Usage( char* argv[] )
{
   fprintf(stderr, "usage: %s <particle-cnt> <timestep-cnt> <timestep-size> <output-freq> <g|i>\n", argv[0]);
   fprintf(stderr, "   'g': initial state of the particles are predefined (generated)\n");
   fprintf(stderr, "   'i': initial state of the particles is read from standard input\n");
   fflush(stderr);
   exit(0);
}

// read initial state of particles from stdin
void ReadState( Particle part[], const int pcnt )
{
   printf("For each particle enter its <mass>, <x,y> coordinates and <x,y> velocities\n");

   // read all particles from standard input
   for( int i = 0, tmp; i < pcnt; )
   {
      tmp = 0;

      printf("part[%2d].m=", i); tmp += scanf("%lf", &part[ i ].m    );

      printf("part[%2d].sx=", i); tmp += scanf("%lf", &part[ i ].s[ X ] );
      printf("part[%2d].sy=", i); tmp += scanf("%lf", &part[ i ].s[ Y ] );

      printf("part[%2d].vx=", i); tmp += scanf("%lf", &part[ i ].v[ X ] );
      printf("part[%2d].vy=", i); tmp += scanf("%lf", &part[ i ].v[ Y ] );

      if( tmp == 5 ) { i++; continue; }
      printf("Invalid particle values -- please reenter\n");
   }
}

// generate initial state of particles
void MakeState( Particle part[], const int pcnt )
{
   double mass = 5.0e24;
   double gap = 1.0e5;
   double speed = 3.0e4;

   // set all particles' variables
   for( int i = 0; i < pcnt; i++ )
   {
      part[ i ].m = mass;

      part[ i ].s[ X ] = i * gap;
      part[ i ].s[ Y ] = 0;

      part[ i ].v[ X ] = 0;
      part[ i ].v[ Y ] = ( 1 - ( ( i&1 )<<1 ) ) * speed;
   }
}

// copy the particles' state from the image into the given particle array
void CopyState( Particle part[], const Particle partcpy[], const int pcnt )
{
   // set all particles' variables from the image
   for( int i = 0; i < pcnt; i++ )
   {
      part[ i ].m      = partcpy[ i ].m;

      part[ i ].s[ X ] = partcpy[ i ].s[ X ];
      part[ i ].s[ Y ] = partcpy[ i ].s[ Y ];

      part[ i ].v[ X ] = partcpy[ i ].v[ X ];
      part[ i ].v[ Y ] = partcpy[ i ].v[ Y ];
   }
}


// print state of particles to output file
void PrintState( FILE* fout, const Particle part[], const int pcnt, const double Ekin, const double Epot, const double tsim, unsigned* _hash, int verbose )
{
   // variable used to calculate the hash function
   // http://www.cse.yorku.ca/~oz/hash.html
   // the starting value is a magic constant
   unsigned hash = 5381;

   // hash the current state
   {
      // size of the string used for hashing
      const int strsz = 128;

      // string used for hashing
      char str[ 1+strsz ];
      str[ strsz ] = '\0';

      // for all particles
      for( int i = 0; i < pcnt; i++ )
      {
         // print the state of the particle to the temporary string
         snprintf( str, strsz, "%13.5e %13.5e %13.5e %13.5e %13.5e\n", part[i].m, part[i].s[X], part[i].s[Y], part[i].v[X], part[i].v[Y] );

         // for all characters in the temporary string
         for( int j = 0; str[j]; j++ )
         {
            // add the current char to the hash
            hash = ( ( hash<<5 ) + hash ) ^ str[j];
         }
      }

      // hash in the total kinetic and potential energy as well
      {
         // print the state of the particle to the temporary string
         snprintf( str, strsz, "%13.5e %13.5e\n", Ekin, Epot );

         // for all characters in the temporary string
         for( int j = 0; str[j]; j++ )
         {
            // add the current char to the hash
            hash = ( ( hash<<5 ) + hash ) ^ str[j];
         }
      }

      // save the resulting hash to the given variable
      *_hash = hash;
   }

   // if the log level is verbose
   if( verbose )
   {
      fprintf( fout, "   particles:\n" );
      fprintf( fout, "   ----------------\n" );
      // for all particles
      for( int i = 0; i < pcnt; i++ )
      {
         // print the state of the current particle
         fprintf( fout, "   %-3d %13.5e   %13.5e %13.5e   %13.5e %13.5e\n", i, part[i].m, part[i].s[X], part[i].s[Y], part[i].v[X], part[i].v[Y] );
      }
   }

   fprintf( fout, "   ----------------\n" );
   fprintf( fout, "   kinetic energy   : %13.5eJ\n", Ekin );
   fprintf( fout, "   potential energy : %13.5eJ\n", Epot );
   fprintf( fout, "   total energy     : %13.5eJ\n", Epot + Ekin );
   fprintf( fout, "   ----------------\n" );
   fprintf( fout, "   simulation time  : %5.2fs\n", tsim );
   fprintf( fout, "   state hash       : %10u\n",   hash );
}



// calculate the force between two particles
// +   moralo je ovako a ne kao funkcija jer se ne izoptimizuje poziv funkcije iz koda (ne ukloni se, a trebalo bi da se ukloni)
#define CalcForce( part, i, j, fX, fY ) \
   /* calculate the distance between particles qubed */ \
   double dX = part[ i ].s[ X ] - part[ j ].s[ X ]; \
   double dY = part[ i ].s[ Y ] - part[ j ].s[ Y ]; \
   double len2 = dX*dX + dY*dY; \
   double len3 = len2 * sqrt( len2 ); \
   \
   /* calculate the magnitude of the force acting on both particles */ \
   double f = ( -G / len3 ) * part[ i ].m * part[ j ].m; \
   fX = f * dX; \
   fY = f * dY;

// update the current particle
// +   moralo je ovako a ne kao funkcija jer se ne izoptimizuje poziv funkcije iz koda (ne ukloni se, a trebalo bi da se ukloni)
#define UpdateParticle( part, i, fX, fY, dt ) \
   /* update the position of the current particle */ \
   part[ i ].s[ X ] += dt * part[ i ].v[ X ]; \
   part[ i ].s[ Y ] += dt * part[ i ].v[ Y ]; \
   /* update the speed of the current particle */ \
   part[ i ].v[ X ] += fX / part[ i ].m  *  dt; \
   part[ i ].v[ Y ] += fY / part[ i ].m  *  dt;

// calculate the kinetic energy increment from the current particle
// +   moralo je ovako a ne kao funkcija jer se ne izoptimizuje poziv funkcije iz koda (ne ukloni se, a trebalo bi da se ukloni)
#define UpdateKinetic( part, i, Einc ) \
   /* calculate the speed of the current particle squared */ \
   double v2 = part[ i ].v[ X ] * part[ i ].v[ X ]  \
             + part[ i ].v[ Y ] * part[ i ].v[ Y ]; \
   \
   /* calculate the increment to the kinetic energy of the system */ \
   double Einc = part[ i ].m * v2 / 2;

// calculate the potential energy increment from the current particle
// +   moralo je ovako a ne kao funkcija jer se ne izoptimizuje poziv funkcije iz koda (ne ukloni se, a trebalo bi da se ukloni)
#define UpdatePotential( part, i, j, Einc ) \
   /* calculate the distance between particles qubed */ \
   double dX = part[ i ].s[ X ] - part[ j ].s[ X ]; \
   double dY = part[ i ].s[ Y ] - part[ j ].s[ Y ]; \
   double len = sqrt( dX*dX + dY*dY ); \
   \
   /* calculate the increment to the potential energy of the system */ \
   double Einc = -G * part[ i ].m * part[ j ].m / len;



// sequential implementation of the n body problem
void Seq( Particle part[], int pcnt, double* _Ekin, double* _Epot, double* _t, double dt, int iters, float* time )
{
   // start the timer
   *time = omp_get_wtime();
   
   // allocate memory used for holding the cumulative force acting on each particle
   Vect2D *const force = ( Vect2D* )malloc( pcnt*sizeof( Vect2D ) );
   // initialize the forces vector
   for( int i = 0; i < pcnt; i++ )
   {
      force[ i ][ X ] = 0;
      force[ i ][ Y ] = 0;
   }

   // update the system the given number of times
   for( int iter = 0; iter < iters; iter++ )
   {
      // calculate the forces acting on each particle
      // for all particles after the current one, add their effect to the current particle
      for( int i = 0; i < pcnt-1; i++ )
      for( int j = i+1; j < pcnt; j++ )
      {
         double fX = 0;
         double fY = 0;
         CalcForce( part, i, j, fX, fY );
         // add the force on each particle to their cumulative forces
         force[ i ][ X ] += fX;   force[ j ][ X ] -= fX;
         force[ i ][ Y ] += fY;   force[ j ][ Y ] -= fY;
      }

      // update all particles
      for( int i = 0; i < pcnt; i++ )
      {
         double fX = force[ i ][ X ];
         double fY = force[ i ][ Y ];
         UpdateParticle( part, i, fX, fY, dt );
         // reset the forces vector
         force[ i ][ X ] = 0;
         force[ i ][ Y ] = 0;
      }
   }


   // temporary kinetic and potential energies of the entire system
   double Ekin = 0;
   double Epot = 0;
   
   // calculate the kinetic energy of the system
   for( int i = 0; i < pcnt; i++ )
   {
      UpdateKinetic( part, i, Einc );
      // add the increment to the kinetic energy of the system
      Ekin += Einc;
   }

   // calculate the potential energy of the system
   // for all particles after the current one, add their effect to the potential energy of the system
   for( int i = 0; i < pcnt-1; i++ )
   for( int j = i+1; j < pcnt; j++ )
   {
      UpdatePotential( part, i, j, Einc );
      // increment the potential energy of the system
      Epot += Einc;
   }

   // save the kinetic and potential energies of the system
   *_Ekin = Ekin;
   *_Epot = Epot;
   // save the simulation time
   *_t = iters*dt;

   // save the execution time
   *time = omp_get_wtime() - *time;

   // free allocated memory
   free( force );
}


// cuda kernel for the parallel implementation
// +   array arguments are coallesced into one array: part = ( mass, posX, posY, speedX, speedY )
// +   two such arrays are needed -- the next array which holds the calculations for the current iteration is swapped with the original array in the next iteration
// +   the result is always saved to the original array
__global__ void kernel( double* part_curr_gpu, double* part_next_gpu, int pcnt, int iters, double dt )
{
   // ...... ++++++ ...... ... ......   <- particles A in this thread block
   //       / |   \  \ 
   // ++++++ ...... ...... ... ......   <- particles B1 interacting with particles A in step 1
   // ...... ++++++ ...... ... ......      particles B2 interacting with particles A in step 2
   // ...... ...... ++++++ ... ......      particles B3 interacting with particles A in step 3
   // ...
   // ...... ...... ...... ... ++++++      particles Bn interacting with particles A in step n

   // the mass, position, calculated speed and calculated force on the particle this thread maps onto
   double massA {};
   double posAx {}, posAy {};
   double spdAx {}, spdAy {};
   double frcAx {}, frcAy {};
   // the mass and position of the particle tile interacting with the particles in this thread block
   // +   stored in shared memory for faster access
   __shared__ double massB[tilesz];
   __shared__ double posBx[tilesz];
   __shared__ double posBy[tilesz];
   
   
   // update the system the given number of times
   for( int iter = 0;   iter < iters;   iter++ )
   {
      // number of tiles in the input particle array(s)
      const int tiles = ceil( float( pcnt )/tilesz );

      // map the current thread block to a <particle tile A> with stride equal to the number of thread blocks in the grid
      for( int tileA = blockIdx.x;   tileA < tiles;   tileA += gridDim.x )
      {
         // the id of the current element this thread maps onto
         int idA = tileA*tilesz + threadIdx.x;
         // if the current element is out of bounds of the global particle array, stop the iteration
         if( idA >= pcnt ) break;


         // initialize the particles in this thread block's tile A from the current global particle array
         {
            // rename parts of the global particle array that hold different types of information
            double *const mass_gpu = part_curr_gpu;
            double *const posx_gpu = &mass_gpu[pcnt];
            double *const posy_gpu = &posx_gpu[pcnt];
            double *const spdx_gpu = &posy_gpu[pcnt];
            double *const spdy_gpu = &spdx_gpu[pcnt];

            // initialize the current particle this thread maps onto in the current tile A this thread block maps onto
            massA = mass_gpu[ idA ];
            posAx = posx_gpu[ idA ]; posAy = posy_gpu[ idA ];
            spdAx = spdx_gpu[ idA ]; spdAy = spdy_gpu[ idA ];
            frcAx = 0;               frcAy = 0;
         }


         // for all tiles B that interact with this thread block's tile A
         for( int tileB = 0;   tileB < tiles;   tileB++ )
         {
            // initialize shared memory used for storing the current tile B
            {
               // the id of the element in tile B in the global array this thread maps onto
               int idB = tileB*tilesz + threadIdx.x;

               // rename parts of the global particle array that hold different types of information
               double *const mass_gpu = part_curr_gpu;
               double *const posx_gpu = &mass_gpu[pcnt];
               double *const posy_gpu = &posx_gpu[pcnt];
               
               // initialize the current tile B in shared memory
               massB[ threadIdx.x ] = mass_gpu[ idB ];
               posBx[ threadIdx.x ] = posx_gpu[ idB ];   posBy[ threadIdx.x ] = posy_gpu[ idB ];

               // wait until all threads in this block have finished initializing their part of the shared memory
               __syncthreads();
            }

            // calculate the forces acting on the <particle this thread maps onto> from interacting with the <particles in tile B>
            for( int j = 0;   j < tilesz;   j++ )
            {
               // the id of the element in tile B in the global array that is currently being used
               int idB = tileB*tilesz + j;
               // if the element B's id is out of bounds of the global particle array, stop the iteration
               if( idB >= pcnt ) break;

               // if the particle is to interact with itself, skip this iteration
               if( idA == idB ) continue;

               // calculate the distance between particles qubed
               double dx = posAx - posBx[j];
               double dy = posAy - posBy[j];
               double len2 = dx*dx + dy*dy;
               double len3 = len2 * sqrt( len2 );
               // calculate the magnitude of the current force acting on the particle
               double f = ( -G/len3 ) * massA*massB[j];

               // update the particle's cumulative force
               frcAx += f * dx;
               frcAy += f * dy;
            }
         }


         // save the particles in this thread block's tile A to the next global particle array
         {
            // rename parts of the next global particle array that hold different types of information
            double *const mass_gpu = part_next_gpu;
            double *const posx_gpu = &mass_gpu[pcnt];
            double *const posy_gpu = &posx_gpu[pcnt];
            double *const spdx_gpu = &posy_gpu[pcnt];
            double *const spdy_gpu = &spdx_gpu[pcnt];

            // update the position of the current particle
            posAx += dt * spdAx;
            posAy += dt * spdAy;
            // update the speed of the current particle
            spdAx += frcAx/massA * dt;
            spdAy += frcAy/massA * dt;

            // save the updated particle to the next global particle array
            mass_gpu[ idA ] = massA;
            posx_gpu[ idA ] = posAx; posy_gpu[ idA ] = posAy;
            spdx_gpu[ idA ] = spdAx; spdy_gpu[ idA ] = spdAy;
         }
      }

      // swap the current and next global particle array pointers
      { double* temp = part_curr_gpu; part_curr_gpu = part_next_gpu; part_next_gpu = temp; }

      // all threads in this grid should finish calculating the current simulation iteration
      cooperative_groups::this_grid().sync();
   }

   // if the final result is located in the next global array of particles, copy it to the original array of particles
   // +   the number of iterations is odd when this happens
   if( iters&1 )
   {
      // number of tiles in the original particle array consisting of five joined arrays
      const int tiles = ceil( float( 5*pcnt )/tilesz );

      // map the current thread block to a <particle tile A> with stride equal to the number of thread blocks in the grid
      for( int tileA = blockIdx.x;   tileA < tiles;   tileA += gridDim.x )
      {
         // the id of the current element this thread maps onto
         int idA = tileA*tilesz + threadIdx.x;
         // if the current element is out of bounds of the global particle array, skip this iteration
         if( idA >= 5*pcnt ) break;
         // save the final result into the original array of particles
         // +   the current array pointer always points to the final result at the beginning of the iteration
         part_next_gpu[ idA ] = part_curr_gpu[ idA ];
      }
   }
}

// parallel implementation of the n body problem
void Par( Particle part[], int pcnt, double* _Ekin, double* _Epot, double* _t, double dt, int iters, float* _time, float* ktime )
{
   // initialize the host and gpu timers
   *_time = 0; *ktime = 0;
   // temporary timer used for updating the host (cumulative) timer
   float time;



   // allocate memory for particles
   // +   group the input particle array's fields together into a new contiguous array
   // +   don't time the conversion between data structures, since it isn't really a part of the algorithm
   double* part_curr = ( double* )malloc( 5*pcnt * sizeof( double ) );

   // initialize the newly created particle array
   {
      // rename parts of the particle array that hold different types of information
      double *const mass = part_curr;
      double *const posx = &mass[pcnt];
      double *const posy = &posx[pcnt];
      double *const spdx = &posy[pcnt];
      double *const spdy = &spdx[pcnt];

      // initialize the new particle array
      for( int i = 0; i < pcnt; i++ )
      {
         mass[i] = part[i].m;
         posx[i] = part[i].s[ X ];
         posy[i] = part[i].s[ Y ];
         spdx[i] = part[i].v[ X ];
         spdy[i] = part[i].v[ Y ];
      }
   }


   
   // start the timer
   time = omp_get_wtime();

   // blosum matrix, sequences which will be compared and the score matrix stored in gpu global memory
   double *part_curr_gpu, *part_next_gpu;
   // allocate space in the gpu global memory
   cudaMalloc( &part_curr_gpu, 5*pcnt * sizeof( double ) );
   cudaMalloc( &part_next_gpu, 5*pcnt * sizeof( double ) );
   // copy data from host to device
	cudaMemcpy( part_curr_gpu, part_curr, 5*pcnt * sizeof( double ), cudaMemcpyHostToDevice );
   // create events for measuring kernel execution time
   cudaEvent_t start, stop;
   cudaEventCreate( &start );
   cudaEventCreate( &stop );
   
   // launch the kernel
   {
      // grid and block dimensions for the kernel
      dim3 grid;
      dim3 block;
      
      // calculate grid and block dimensions for the kernel
      {
         // take the number of particles in a tile as the number of threads in a block
         block.x = tilesz;
         // take the number of tiles in the particle array as the initial number of thread blocks in the grid
         grid.x = ceil( float( pcnt )/tilesz );

         // the maximum number of parallel blocks on a streaming multiprocessor
         int maxBlocksPerSm = 0;
         // number of threads per block that the kernel will be launched with
         int numThreads = tilesz;
         // size of shared memory per block in bytes
         int sharedMemSz = ( 3*tilesz )*sizeof( double );

         // calculate the max number of parallel blocks per streaming multiprocessor
         cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxBlocksPerSm, kernel, numThreads, sharedMemSz );
         // the number of cooperative blocks launched must not exceed the maximum possible number of parallel blocks on the device
         grid.x = min( grid.x, MPROCS*maxBlocksPerSm );
      }

      // group arguments to be passed to the kernel
      void* kargs[] { &part_curr_gpu, &part_next_gpu, &pcnt, &iters, &dt };
      
      // launch the kernel in the given stream (don't statically allocate shared memory)
      // +   capture events around kernel launch as well
      // +   update the stop event when the kernel finishes
      cudaEventRecord( start, STREAM_ID );
      cudaLaunchCooperativeKernel( ( void* )kernel, grid, block, kargs, 0, STREAM_ID );
      cudaEventRecord( stop, STREAM_ID );
      cudaEventSynchronize( stop );
      
      // calculate the time between the given events
      // +   update the total kernel execution time
      cudaEventElapsedTime( ktime, start, stop );
   }

   // wait for the gpu to finish before going to the next step
   cudaDeviceSynchronize();
   // save the updated particles into the temporary array
   // +   waits for the device to finish, then copies data from device to host
   cudaMemcpy( part_curr, part_curr_gpu, 5*pcnt * sizeof( double ), cudaMemcpyDeviceToHost );

   // stop the timer
   time = ( omp_get_wtime() - time );
   // add the time to the cumulative host timer
   *_time += time;



   // update the original particle array
   // +   don't time the conversion between data structures, since it isn't really a part of the algorithm
   {
      // rename parts of the newly created particle array that hold different types of information
      double *const mass = part_curr;
      double *const posx = &mass[pcnt];
      double *const posy = &posx[pcnt];
      double *const spdx = &posy[pcnt];
      double *const spdy = &spdx[pcnt];

      // update the original particle array
      for( int i = 0; i < pcnt; i++ )
      {
         part[i].m      = mass[i];
         part[i].s[ X ] = posx[i];
         part[i].s[ Y ] = posy[i];
         part[i].v[ X ] = spdx[i];
         part[i].v[ Y ] = spdy[i];
      }

      // free allocated memory
      free( part_curr );
   }



   // start the timer
   time = omp_get_wtime();

   // temporary kinetic and potential energies of the entire system
   double Ekin = 0;
   double Epot = 0;

   // calculate the kinetic energy of the system
   for( int i = 0; i < pcnt; i++ )
   {
      UpdateKinetic( part, i, Einc );
      
      // if the kinetic energy increment is a real number
      if( isfinite( Einc ) )
      {
         // add the increment to the kinetic energy of the system
         Ekin += Einc;
      }
   }

   // calculate the potential energy of the system
   // for all particles after the current one, add their effect to the potential energy of the system
   for( int i = 0; i < pcnt-1; i++ )
   for( int j = i+1; j < pcnt; j++ )
   {
      UpdatePotential( part, i, j, Einc );
      // if the potential energy increment is a real number
      if( isfinite( Epot ) )
      {
         // increment the potential energy of the system
         Epot += Einc;
      }
   }
   
   // save the kinetic and potential energies of the system
   *_Ekin = Ekin;
   *_Epot = Epot;
   // save the simulation time
   *_t = iters*dt;

   // stop the timer
   time = ( omp_get_wtime() - time );
   // add the time to the cumulative host timer
   *_time += time;



   // free allocated space in the gpu global memory
   cudaFree( part_curr_gpu );
   cudaFree( part_next_gpu );
   // free events' memory
   cudaEventDestroy( start );
   cudaEventDestroy( stop );
}





// main program
int main( int argc, char* argv[] )
{
   fflush(stdout);
   if( argc != 6 ) Usage( argv );

   int pcnt  = strtol( argv[1], NULL, 10 );   // number of particles
   int iters = strtol( argv[2], NULL, 10 );   // number of simulation steps
   double dt = strtod( argv[3], NULL );       // size of timestep
   char initmode = argv[5][0];                // initialization mode

   // check if the given arguments are valid
   if( pcnt <= 0 || iters <= 0 || dt <= 0 ) Usage( argv );
   if( initmode != 'g' && initmode != 'i' ) Usage( argv );


   // array of particles in the system, and the copy of the original array of particles in the system
   Particle *const part    = ( Particle* )malloc( pcnt*sizeof( Particle ) );
   Particle *const partcpy = ( Particle* )malloc( pcnt*sizeof( Particle ) );
   // kinetic and potential energy of those particles
   double Ekin = 0, Epot = 0;
   // simulation time
   double t = 0;

   // initialize all the particles in the system
   if( initmode == 'i' ) ReadState( partcpy, pcnt );
   else                  MakeState( partcpy, pcnt );
   // copy the initial state from the image to the current state
   CopyState( part, partcpy, pcnt );


   // hashes of the simulation results
   unsigned hash0 = 10;
   unsigned hash1 = 20;
   unsigned hash2 = 30;

   // calculate initial energy
   {
      // calculate the kinetic energy of the system
      for( int i = 0; i < pcnt; i++ )
      {
         /* calculate the speed of the current particle squared */ \
         double v2 = part[ i ].v[ X ] * part[ i ].v[ X ]  \
                   + part[ i ].v[ Y ] * part[ i ].v[ Y ]; \
         \
         /* calculate the increment to the kinetic energy of the system */ \
         double Einc = part[ i ].m * v2 / 2;
         // add the increment to the kinetic energy of the system
         Ekin += Einc;
      }

      // calculate the potential energy of the system
      // for all particles after the current one, add their effect to the potential energy of the system
      for( int i = 0; i < pcnt-1; i++ )
      for( int j = i+1; j < pcnt; j++ )
      {
         UpdatePotential( part, i, j, Einc );
         // increment the potential energy of the system
         Epot += Einc;
      }

      printf( "Initial state for both implementations\n" );

      // print the system state and simulation stats to stdout
      PrintState( stdout, part, pcnt, Ekin, Epot, t, &hash0, 0 );

      printf( "\n" );
      fflush(stdout);

      // copy the initial state from the image to the current state
      CopyState( part, partcpy, pcnt );
   }

   // sequential implementation
   {
      printf( "Sequential implementation\n" );

      // simulation execution time
      float time;
      // start the sequential implementation
      Seq( part, pcnt, &Ekin, &Epot, &t, dt, iters, &time );

      // print the system state and simulation stats to a file and stdout
      FILE *fout = fopen( "nbody.out1.txt", "w" );
      PrintState( fout,   part, pcnt, Ekin, Epot, t, &hash1, 1 );   fclose( fout );
      PrintState( stdout, part, pcnt, Ekin, Epot, t, &hash1, 0 );
      printf( "   execution time   : %9.6fs\n", time );

      printf( "\n" );
      fflush(stdout);

      // copy the initial state from the image to the current state
      CopyState( part, partcpy, pcnt );
   }

   // parallel implementation
   {
      printf( "Parallel implementation\n" );

      // simulation and kernel execution time
      float time, ktime;
      // start the parallel implementation
      Par( part, pcnt, &Ekin, &Epot, &t, dt, iters, &time, &ktime );

      // print the system state and simulation stats to a file and stdout
      FILE *fout = fopen( "nbody.out2.txt", "w" );
      PrintState( fout,   part, pcnt, Ekin, Epot, t, &hash2, 1 );   fclose( fout );
      PrintState( stdout, part, pcnt, Ekin, Epot, t, &hash2, 0 );
      printf( "   execution time   : %9.6fs   ktime: %9.6fs\n", time, ktime );

      printf( "\n" );
      fflush(stdout);

      // copy the initial state from the image to the current state
      CopyState( part, partcpy, pcnt );
   }

    // compare the implementations
    if( hash1 == hash2 ) printf( "TEST PASSED\n" );
    else                 printf( "TEST FAILED\n" );
    fflush(stdout);

   // free allocated memory
   free( part ); free( partcpy );
}
