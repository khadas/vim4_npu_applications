#ifndef _POSTPROCESS_UTIL_H_
#define _POSTPROCESS_UTIL_H_

#include "nn_demo.h"

void postprocess_densenet_ctc(nn_output *pout, char* result, int* result_len);

void l2_normalize(float* input);

float eu_distance(float* input);

float compare_eu_distance(float* input1, float* input2);

float cos_similarity(float* input1, float* input2);

#endif //POSTPROCESS_UTIL_H
