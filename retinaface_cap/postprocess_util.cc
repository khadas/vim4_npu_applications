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
#include "nn_demo.h"

int g_detect_number = 50; //max detect num

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
    float threshold = 0.5;
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
    retinaface_buffer[0] = (float*)pout->out[1].buf; // 8
    retinaface_buffer[1] = (float*)pout->out[0].buf; // 4
    retinaface_buffer[2] = (float*)pout->out[2].buf; // 20

    retinaface_postprocess(retinaface_buffer,640,640,20,20,0,dectout);
}
