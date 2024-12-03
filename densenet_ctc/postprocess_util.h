#ifndef _POSTPROCESS_UTIL_H_
#define _POSTPROCESS_UTIL_H_

#include "nn_demo.h"

void postprocess_densenet_ctc(nn_output *pout, char* result, int* result_len);


#endif //POSTPROCESS_UTIL_H
