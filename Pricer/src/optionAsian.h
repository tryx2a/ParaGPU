#ifndef _OPTIONASIAN_H
#define _OPTIONASIAN_H

#include "option.h"

class OptionAsian : 
	public Option
{
public:
	double strike_;

	OptionAsian(double T_, int TimeSteps_, int size_, double strike_);

	~OptionAsian();

	double payoff(const PnlMat *path);
};

#endif //_OPTIONASIAN_H
