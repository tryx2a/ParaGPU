#include <cuda.h>
#include "bs.h"
#include "option.h"
#include "optionBasket.h"
#include "mc.h"

class CudaLib
{
public:

	int maxDevice;

	CudaLib();
	~CudaLib();

	void loadOption(Option* opt);
	void loadBS(BS* bs);
	void loadMonteCarlo(MonteCarlo* mc);

	
};
