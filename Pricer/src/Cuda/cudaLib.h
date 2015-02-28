#include <cuda.h>
#include "../Model/bs.h"
#include "../Options/option.h"
#include "../Method/mc.h"

class CudaLib
{
public:
        //Nombre Maximum de Threads
	int maxDevice;
        
        /* Paramètres de BS*/
        float *trend; /// trend des actifs du marché
        float *sigma; /// vecteur de volatilités
        float *spot; /// valeurs initiales du sous-jacent
        float *chol; /// matrice de cholesky calculé dans le constructeur
        
        /* Paramètres Option */
        float *payoffCoeff; /// payoff coefficient
        float *lowerBarrier; /// barrière basse
        float *upperBarrier; /// barrière haute


        /* Paramètres autres */
        float strike; ///Strike de l'option
        float *tabPath; /// Tableau contenant une matrice path par device
        float *tabPrice; /// Tableau contenant le prix calculé par chaque thread
        float *tabVar; /// Tableau contenant la longueur de la variance calculée par chaque thread


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
