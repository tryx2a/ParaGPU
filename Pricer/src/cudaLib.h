#include <cuda.h>
#include "bs.h"
#include "option.h"
#include "mc.h"

class CudaLib
{
public:
        //Nombre Maximum de Threads
	int maxDevice;
        
        /* Paramètre de BS*/
        float *trend; /// trend des actifs du marché
        float *sigma; /// vecteur de volatilités
        float *spot; /// valeurs initiales du sous-jacent
        float *chol; /// matrice de cholesky calculé dans le constructeur
        
        /* Paramètre Option */
        float *payoffCoeff; /// payoff coefficient

        /**
         * Constructeur de CudaLib
         * @param[in] mc : objet de type MonteCarlo permettant
         * d'avoir tous les paramètres nécessaire pour la 
         * valorisation du produit.
         */
	CudaLib(MonteCarlo* mc);

        /**
         * Destructeur de CudaLib
         */
	~CudaLib();


        /**
         * Méthode peremttant d'allouer les objets de types pnl
         * dans le GPU.
         * @param[in] opt
         */
	void allocOption(Option* opt);

        /**
         * Méthode peremttant d'allouer les objets de types pnl
         * dans le GPU.
         * @param[in] bs
         */
	void allocBS(BS* bs);

        /**
         * Méthode peremttant d'allouer les objets de types pnl
         * dans le GPU.
         * @param[in] mc
         */
	void allocMonteCarlo(MonteCarlo* mc);
	
        /**
         * Méthode peremttant de charger en mémoire les objets de types pnl
         * dans le GPU.
         * @param[in] opt
         */
	void memcpyOption(Option* opt);

        /**
         * Méthode peremttant de charger en mémoire les objets de types pnl
         * dans le GPU.
         * @param[in] bs
         */
	void memcpyBS(BS* bs);

        /**
         * Méthode peremttant de charger en mémoire les objets de types pnl
         * dans le GPU.
         * @param[in] mc
         */
	void memcpyMonteCarlo(MonteCarlo* mc);

	
};
