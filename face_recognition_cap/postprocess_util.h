#ifndef _POSTPROCESS_UTIL_H_
#define _POSTPROCESS_UTIL_H_

#include <opencv2/opencv.hpp>
#include "nn_demo.h"

void postprocess_retinaface(nn_output *pout, face_detect_out_t* dectout);

cv::Mat similarTransform(cv::Mat src,cv::Mat dst);

void l2_normalize(float* input);

float eu_distance(float* input);

float compare_eu_distance(float* input1, float* input2);

float cos_similarity(float* input1, float* input2);

#endif //POSTPROCESS_UTIL_H
