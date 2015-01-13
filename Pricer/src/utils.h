#ifndef UTILS_H
#define UTILS_H

#include "pnl/pnl_vector.h"
#include "pnl/pnl_matrix.h"

namespace utils {
  double computePayoff(const PnlMat *path, const PnlVect *payoffCoeff_, int TimeSteps_, double strike);
} // utils


#endif // UTILS_H
