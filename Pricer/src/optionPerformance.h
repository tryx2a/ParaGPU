#ifndef _OPTIONPERFORMANCE_H
#define _OPTIONPERFORMANCE_H

#include "option.h"

class OptionPerformance : public Option
{
public:

  	PnlVect* payoffCoeff_; 
  
	OptionPerformance(double T_, int timeSteps_, int size_, PnlVect* payoffCoeff);
	
	~OptionPerformance();

	double payoff(const PnlMat *path);

};

#endif //_OPTIONPERFORMANCE_H
