#include <cuda.h>
#include "bs.h"
#include "option.h"
#include "optionBasket.h"
#include "mc.h"

class CudaLib
{
public:
        //Nombre Maximum de Threads
	int maxDevice;
	
	/* Paramètre de MonteCarlo*/
	float h; /*! pas de différence finie */
        int H; /* nombre de période de rebalancement*/
        int samples; /*! nombre de tirages Monte Carlo */
        
        /* Paramètre de BS*/
        int size; /// nombre d'actifs du modèle
        float r; /// taux d'intérêt
       
        int size_trend;
        double *trend; /// trend des actifs du marché
       
        float rho; /// paramètre de corrélation
       
        int size_sigma;
        double *sigma; /// vecteur de volatilités
       
        int size_spot;
        double *spot; /// valeurs initiales du sous-jacent
       
        int m;
        int n;
        double *chol; /// matrice de cholesky calculé dans le constructeur
        
        /* Paramètre Option */
        float T; /// maturité
        int TimeSteps; /// nombre de pas de temps de discrétisation
        float strike; /// strike de l'option
        int size_payoffCoeff;
        double *payoffCoeff; /// payoff coefficient

	CudaLib(MonteCarlo* mc);
	~CudaLib();

	void allocOption(Option* opt);
	void allocBS(BS* bs);
	void allocMonteCarlo(MonteCarlo* mc);
	
	void loadOption(Option* opt);
	void loadBS(BS* bs);
	void loadMonteCarlo(MonteCarlo* mc);

	
};
