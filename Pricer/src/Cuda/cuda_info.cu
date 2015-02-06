#include  <cuda.h>
#include <stdio.h>

int main(int argc, char ** argv) {
    int deviceCount;

    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;

        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
               // Ne detecte pas CUDA
                return -1;
            } else {
              // Afficher le nombre de device
              if (deviceCount == 0) {
                printf("There is no device supporting CUDA.\n");
                exit (0);
	      }
	      else{
		printf("Number of device : %d\n",deviceCount);
	      }

            }
        }

        // Afficher le nom de la device
	printf("Device Name : %s\n", deviceProp.name);
	
        // Donner le numero de version majeur et mineur
        printf("Major : %d, Minor : %d\n",deviceProp.major, deviceProp.minor);
        
        // Donner la taille de la memoire globale
        printf("Global Memory : %d\n", deviceProp.totalGlobalMem);
        
        // Donner la taille de la memoire constante
        printf("Constant Memory : %d\n", deviceProp.totalConstMem);
        
        // Donner la taille de la memoire partagee par bloc
        printf("Shared Memory : %d\n", deviceProp.sharedMemPerBlock);
        
        int i = 0;
        for(i = 0; i<4; i++){
                // Donner le nombre de thread max dans chacune des directions
                printf("Number Thread Max %d: %d\n", i, deviceProp.maxThreadsDim[i]);
        
                // Donner le taille maximum de la grille pour chaque direction
                printf("Size Max Grid : %d %d\n", i, deviceProp.maxGridSize[i]);
        }
        
        // Donner la taille du warp
        printf("Warp Size : %d\n", deviceProp.warpSize);
    }

    return 0;
}
