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

static const char *names[] = {
	"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "+", "-", "*", "/", ",", ".", "[", "]", "{", "}", "|", "~", "@", "#", "$", "%", "^", "&", "(", ")", "<", ">", "?", ":", ";", "a", "b", "c", "d", "e", "f", "g", "h", "i", "g", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
};

/** Status enum */
typedef enum
{
    UTIL_FAILURE = -1,
    UTIL_SUCCESS = 0,
}nn_status_e;

/*-------------------------------------------
                  Functions
-------------------------------------------*/

void postprocess_densenet_ctc(nn_output *pout, char* result, int* result_len)
{
    float* densenet_ctc_buffer[1] = {NULL};
    densenet_ctc_buffer[0] = (float*)pout->out[0].buf;
    
    int i, j, max_index;
    int box = 35;
    int class_num = 88;
    float threshold = 0.25;
    float max_conf, conf;
    
    int last_index = class_num - 1;
    for (i = 0; i < box; ++i)
    {
    	max_conf = 0;
    	max_index = class_num - 1;
    	for (j = 0; j < class_num - 1; ++j)
    	{
    		conf = densenet_ctc_buffer[0][i * class_num + j];
    		if (conf > threshold && conf > max_conf)
    		{	
    			max_conf = conf;
    			max_index = j;
    		}
    	}
    	if (max_index != class_num - 1 && max_index != last_index)
    	{
    		result[*result_len] = *names[max_index];
    		(*result_len)++;
    	}
    	last_index = max_index;
    }
}
