/****************************************************************************
*
*    Copyright (c) 2019  by amlogic Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of amlogic Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of amlogic Corporation.
*
***************************************************************************/
/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <fcntl.h>
#include "nn_util.h"
#include "nn_sdk.h"
#include <opencv2/opencv.hpp>
//#include "nn_demo.h"

/** Status enum */
typedef enum
{
    UTIL_FAILURE = -1,
    UTIL_SUCCESS = 0,
}nn_status_e;

/*-------------------------------------------
                  Functions
-------------------------------------------*/

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float area = 0;
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0)
        return 0;
    area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

float softmax(float input1, float input2)
{
	float total = exp(input1) + exp(input2);
	return exp(input2) / total;
}

void retinaface_postprocess(float **predictions, int width, int height, int modelWidth, int modelHeight, int input_num, face_detect_out_t* dectout)
{
    int nn_width,nn_height, nn_channel;
    void* objout = NULL;
    nn_width = 640;
    nn_height = 640;
    nn_channel = 3;
    (void)nn_channel;
    float min_sizes[3][2] = {{16, 32}, {64, 128}, {256, 512}};
    float variance[2] = {0.1, 0.2};
    int grid[3] = {80, 40, 20};
    int result_len = 0;
    int total_num = grid[0] * grid[0] + grid[1] * grid[1] + grid[2] * grid[2];

    int i, j, k, index;
    int initial = 0;
    float threshold = 0.7;
    float iou_threshold = 0.1;
    float conf;
    
    float *box_predictions = predictions[0];
    float *conf_predictions = predictions[1];
    float *point_predictions = predictions[2];

    box *boxes = (box *)calloc(total_num * 2, sizeof(box));
    float *probs = (float *)calloc(total_num * 2, sizeof(float));
    float **points = (float **)calloc(total_num * 2, sizeof(float *));

    for (i = 0; i < total_num * 2; ++i)
    {
    	points[i] = (float *)calloc(10, sizeof(float));
    }

    for (int n = 0; n < 3; n++)
    {
    	if (n == 1)
    	{
    		initial = pow(grid[0], 2);
    	}
    	else if (n == 2)
    	{
    		initial = pow(grid[0], 2) + pow(grid[1], 2);
    	}
    	for (i = 0; i < grid[n]; ++i)
    	{
    		for (j = 0; j < grid[n]; ++j)
    		{
    			for (k = 0; k < 2; ++k)
    			{
    				index = initial + i * grid[n] + j;
    				conf = softmax(conf_predictions[index + 2 * k * total_num], conf_predictions[index + (2 * k + 1) * total_num]);
    				if (conf >= threshold)
    				{
    					boxes[result_len].x = (j + 0.5) / grid[n] + box_predictions[index + (4 * k + 0) * total_num] * variance[0] * min_sizes[n][k] / nn_width;
    					boxes[result_len].y = (i + 0.5) / grid[n] + box_predictions[index + (4 * k + 1) * total_num] * variance[0] * min_sizes[n][k] / nn_height;
    					boxes[result_len].w = min_sizes[n][k] / nn_width * exp(box_predictions[index + (4 * k + 2) * total_num] * variance[1]);
    					boxes[result_len].h = min_sizes[n][k] / nn_height * exp(box_predictions[index + (4 * k + 3) * total_num] * variance[1]);
    					
    					points[result_len][0] = (j + 0.5) / grid[n] + point_predictions[index + (10 * k + 0) * total_num] * variance[0] * min_sizes[n][k] / nn_width;
    					points[result_len][1] = (i + 0.5) / grid[n] + point_predictions[index + (10 * k + 1) * total_num] * variance[0] * min_sizes[n][k] / nn_height;
    					points[result_len][2] = (j + 0.5) / grid[n] + point_predictions[index + (10 * k + 2) * total_num] * variance[0] * min_sizes[n][k] / nn_width;
    					points[result_len][3] = (i + 0.5) / grid[n] + point_predictions[index + (10 * k + 3) * total_num] * variance[0] * min_sizes[n][k] / nn_height;
    					points[result_len][4] = (j + 0.5) / grid[n] + point_predictions[index + (10 * k + 4) * total_num] * variance[0] * min_sizes[n][k] / nn_width;
    					points[result_len][5] = (i + 0.5) / grid[n] + point_predictions[index + (10 * k + 5) * total_num] * variance[0] * min_sizes[n][k] / nn_height;
    					points[result_len][6] = (j + 0.5) / grid[n] + point_predictions[index + (10 * k + 6) * total_num] * variance[0] * min_sizes[n][k] / nn_width;
    					points[result_len][7] = (i + 0.5) / grid[n] + point_predictions[index + (10 * k + 7) * total_num] * variance[0] * min_sizes[n][k] / nn_height;
    					points[result_len][8] = (j + 0.5) / grid[n] + point_predictions[index + (10 * k + 8) * total_num] * variance[0] * min_sizes[n][k] / nn_width;
    					points[result_len][9] = (i + 0.5) / grid[n] + point_predictions[index + (10 * k + 9) * total_num] * variance[0] * min_sizes[n][k] / nn_height;
    					
    					boxes[result_len].prob_obj = conf;
    					
    					result_len++;
    				}
    			}
    		}
    	}
    }
    
    dectout->pBox = (detBox*)malloc(result_len*sizeof(detBox));
    dectout->point_1 = (point_t*)malloc(result_len*sizeof(point_t));
    dectout->point_2 = (point_t*)malloc(result_len*sizeof(point_t));
    dectout->point_3 = (point_t*)malloc(result_len*sizeof(point_t));
    dectout->point_4 = (point_t*)malloc(result_len*sizeof(point_t));
    dectout->point_5 = (point_t*)malloc(result_len*sizeof(point_t));
    
    index = 0;
    for (i = 0; i < result_len; ++i)
    {
    	if (boxes[i].prob_obj != -1)
    	{
    		for (j = i + 1; j < result_len; ++j)
    		{
    			if (boxes[j].prob_obj != -1)
    			{
    				float iou = box_iou(boxes[i], boxes[j]);
    				
    				if (iou > iou_threshold)
    				{
    					if (boxes[i].prob_obj >= boxes[j].prob_obj)
    					{
    						boxes[j].prob_obj = -1;
    					}
    					else
    					{
    						boxes[i].prob_obj = -1;
    						break;
    					}
    				}
    			}
    		}
    	}
    	if (boxes[i].prob_obj != -1)
    	{
    		dectout->pBox[index].x = boxes[i].x;
    		dectout->pBox[index].y = boxes[i].y;
    		dectout->pBox[index].w = boxes[i].w;
    		dectout->pBox[index].h = boxes[i].h;
    		dectout->pBox[index].score = boxes[i].prob_obj;
    		dectout->pBox[index].objectClass = 0.0;
    		//printf("%f %f %f %f\n", boxes[i].x, boxes[i].y, boxes[i].w, boxes[i].h);
    		
    		dectout->point_1[index].x = points[i][0];
    		dectout->point_1[index].y = points[i][1];
    		dectout->point_2[index].x = points[i][2];
    		dectout->point_2[index].y = points[i][3];
    		dectout->point_3[index].x = points[i][4];
    		dectout->point_3[index].y = points[i][5];
    		dectout->point_4[index].x = points[i][6];
    		dectout->point_4[index].y = points[i][7];
    		dectout->point_5[index].x = points[i][8];
    		dectout->point_5[index].y = points[i][9];
    		
    		index++;
    	}
    }
    dectout->detNum = index;
    
    free(boxes);
    boxes = NULL;
    free(probs);
    probs = NULL;
    for (i = 0; i < total_num * 2; ++i)
    {
    	free(points[i]);
    }
    free(points);
    points = NULL;
}

void postprocess_retinaface(nn_output *pout, face_detect_out_t* dectout)
{
    float *retinaface_buffer[3] = {NULL};
    retinaface_buffer[0] = (float*)pout->out[1].buf;
    retinaface_buffer[1] = (float*)pout->out[2].buf;
    retinaface_buffer[2] = (float*)pout->out[0].buf;

    retinaface_postprocess(retinaface_buffer,640,640,20,20,0,dectout);
}

cv::Mat meanAxis0(const cv::Mat &src)
{
        int num = src.rows;
        int dim = src.cols;
 
        // x1 y1
        // x2 y2
 
        cv::Mat output(1,dim,CV_32F);
        for(int i = 0 ; i <  dim; i ++)
        {
            float sum = 0 ;
            for(int j = 0 ; j < num ; j++)
            {
                sum+=src.at<float>(j,i);
            }
            output.at<float>(0,i) = sum/num;
        }
 
        return output;
}

cv::Mat elementwiseMinus(const cv::Mat &A,const cv::Mat &B)
{
        cv::Mat output(A.rows,A.cols,A.type());
 
        assert(B.cols == A.cols);
        if(B.cols == A.cols)
        {
            for(int i = 0 ; i <  A.rows; i ++)
            {
                for(int j = 0 ; j < B.cols; j++)
                {
                    output.at<float>(i,j) = A.at<float>(i,j) - B.at<float>(0,j);
                }
            }
        }
        return output;
}

cv::Mat varAxis0(const cv::Mat &src)
{
	cv::Mat temp_ = elementwiseMinus(src,meanAxis0(src));
        cv::multiply(temp_ ,temp_ ,temp_ );
        return meanAxis0(temp_);
 
}

int MatrixRank(cv::Mat M)
{
	cv::Mat w, u, vt;
	cv::SVD::compute(M, w, u, vt);
	cv::Mat1b nonZeroSingularValues = w > 0.0001;
        int rank = countNonZero(nonZeroSingularValues);
        return rank;
 
}

cv::Mat similarTransform(cv::Mat src,cv::Mat dst) {
        int num = src.rows;
        int dim = src.cols;
        cv::Mat src_mean = meanAxis0(src);
        cv::Mat dst_mean = meanAxis0(dst);
        cv::Mat src_demean = elementwiseMinus(src, src_mean);
        cv::Mat dst_demean = elementwiseMinus(dst, dst_mean);
        cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
        cv::Mat d(dim, 1, CV_32F);
        d.setTo(1.0f);
        if (cv::determinant(A) < 0) {
            d.at<float>(dim - 1, 0) = -1;
        }
	cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
        cv::Mat U, S, V;
	cv::SVD::compute(A, S,U, V);
        // the SVD function in opencv differ from scipy .
        int rank = MatrixRank(A);
        if (rank == 0) {
            assert(rank == 0);
        } else if (rank == dim - 1) {
            if (cv::determinant(U) * cv::determinant(V) > 0) {
                T.rowRange(0, dim).colRange(0, dim) = U * V;
            } else {
//            s = d[dim - 1]
//            d[dim - 1] = -1
//            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
//            d[dim - 1] = s
                int s = d.at<float>(dim - 1, 0) = -1;
                d.at<float>(dim - 1, 0) = -1;
                T.rowRange(0, dim).colRange(0, dim) = U * V;
                cv::Mat diag_ = cv::Mat::diag(d);
                cv::Mat twp = diag_*V; //np.dot(np.diag(d), V.T)
		cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
		cv::Mat C = B.diag(0);
                T.rowRange(0, dim).colRange(0, dim) = U* twp;
                d.at<float>(dim - 1, 0) = s;
            }
        }
        else{
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_*V.t(); //np.dot(np.diag(d), V.T)
            cv::Mat res = U* twp; // U
            T.rowRange(0, dim).colRange(0, dim) = -U.t()* twp;
        }
        cv::Mat var_ = varAxis0(src_demean);
        float val = cv::sum(var_).val[0];
        cv::Mat res;
        cv::multiply(d,S,res);
        float scale =  1.0/val*cv::sum(res).val[0];
        T.rowRange(0, dim).colRange(0, dim) = - T.rowRange(0, dim).colRange(0, dim).t();
        cv::Mat  temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
        cv::Mat  temp2 = src_mean.t(); //src_mean.T
        cv::Mat  temp3 = temp1*temp2; // np.dot(T[:dim, :dim], src_mean.T)
        cv::Mat temp4 = scale*temp3;
        T.rowRange(0, dim).colRange(dim, dim+1)=  -(temp4 - dst_mean.t()) ;
        T.rowRange(0, dim).colRange(0, dim) *= scale;
        return T;
}

void l2_normalize(float* input)
{
	float sum = 0;
	for (int i = 0; i < 128; ++i)
	{
		sum = sum + input[i] * input[i];
	}
	sum = sqrt(sum);
	for (int i = 0; i < 128; ++i)
	{
		input[i] = input[i] / sum;
	}
}

float eu_distance(float* input)
{
	float sum = 0;
	for (int i = 0; i < 128; ++i)
	{
		sum = sum + input[i] * input[i];
	}
	sum = sqrt(sum);
	return sum;
}

float compare_eu_distance(float* input1, float* input2)
{
	float sum = 0;
	for (int i = 0; i < 128; ++i)
	{
		sum = sum + (input1[i] - input2[i]) * (input1[i] - input2[i]);
	}
	sum = sqrt(sum);
	return sum;
}

float cos_similarity(float* input1, float* input2)
{
	float sum = 0;
	for (int i = 0; i < 128; ++i)
	{
		sum = sum + input1[i] * input2[i];
	}
	float tmp1 = eu_distance(input1);
	float tmp2 = eu_distance(input2);
	return sum / (tmp1 * tmp2);
}
