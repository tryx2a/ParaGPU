#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


int n = 200;
using namespace std;

__device__ float generate( curandState* globalState, int ind ) 
{
    int ind = threadIdx.x;
    curandState localState = globalState[ind];
    //float RANDOM = curand_uniform( &localState );
    float RANDOM = curand_normal( &localState );
    globalState[ind] = localState;
    return RANDOM;
}

__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
}

__global__ void kernel(float* tabGauss, curandState* globalState, int n, int nbTours)
{
    // generate random numbers
    for(int i=0;i<nbTours;i++)
    {
        float k = generate(globalState, i) * 1;
        
        /*while(k > n*n-1)
        {
            k-=(n*n-1);
        }*/

        tabGauss[i] = k;
    }
}


int main() 
{
    int N=15;

    curandState* devStates;
    cudaMalloc ( &devStates, N*sizeof( curandState ) );

    // setup seeds
    setup_kernel <<< 1, N >>> ( devStates,unsigned(time(NULL)) );

    float N2[N];
    float* N3;
    cudaMalloc((void**) &N3, sizeof(float)*N);

    kernel<<<1,1>>> (N3, devStates, n, N);

    cudaMemcpy(N2, N3, sizeof(float)*N, cudaMemcpyDeviceToHost);

    for(int i=0;i<N;i++)
    {
        cout<<N2[i]<<endl;
    }



    return 0;
}