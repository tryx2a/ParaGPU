#include "utils.h"
#include <assert.h> 

namespace utils {

  double computePayoff(const PnlMat *path, const PnlVect *payoffCoeff_, int TimeSteps_, double strike){
    double res = 0.0;

    int i;
    PnlVect * assetAtMaturity = pnl_vect_new();

    //On extrait la ligne 
    pnl_mat_get_row(assetAtMaturity, path, TimeSteps_); 
    
    assert(assetAtMaturity->size == payoffCoeff_->size);

    for(int i = 0; i<assetAtMaturity->size; i++){
      res += GET(assetAtMaturity,i)*GET(payoffCoeff_,i);
    }

    pnl_vect_free(&assetAtMaturity);

    res -= strike;

    if(res<=0.0){
      return 0.0;
    }

    return res;
  }

} // utils
