#include <iostream>

using namespace std;

#include "optionPerformance.h"

/**
 * Constructeur de la classe OptionPerformance
 */
OptionPerformance::OptionPerformance(double T_, int timeSteps_, int size_, PnlVect* payoffCoeff): Option(T_, timeSteps_, size_)
{
	this->payoffCoeff_ = payoffCoeff;
	this->id_=6;
}

/**
 *	Destructeur de la classe OptionPerformance
 */
 OptionPerformance::~OptionPerformance(){
 	pnl_vect_free(&(this->payoffCoeff_));
 }

/**
 * Implémentation de la méthode de payoff
 */
 double OptionPerformance::payoff(const PnlMat *path){
 	
 	double ratio = 0.0;
 	double num = 0.0; 
 	double den = 0.0;

 	PnlVect *assetAtToday = pnl_vect_new();
 	PnlVect *assetAtYesterday = pnl_vect_new();

 	//Sum coeff ratio
 	for(int i = 1 ; i <= this->TimeSteps_ ; i++){

 		//On extrait la ligne a t=t_i
  		pnl_mat_get_row(assetAtToday, path, i);
  		//On extract la ligne a t=t_i-1 
  		pnl_mat_get_row(assetAtYesterday, path, i-1); 
  
 		//Compute numerator and denom
 		for(int d = 0 ; d < this->size_ ; d++){
 			
 			num += GET(assetAtToday,d)*GET(payoffCoeff_,d);
 			den += GET(assetAtYesterday,d)*GET(payoffCoeff_,d);
 			
 		}
 		//Compute of ratio
 		ratio += num/den; 
 	}

 	ratio = ratio/this->TimeSteps_ - 1;
	
	//Refresh ratio
 	if(ratio < 0){
 		ratio = 0;
 	}
 	
 	 pnl_vect_free(&assetAtToday);
 	 pnl_vect_free(&assetAtYesterday);

 	//Compute min
 	if(ratio > 0.1){
 		return 1.1;
 	}else{
 		return 1 + ratio;
 	}
 	
 	
 }
