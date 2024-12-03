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
	"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck", "no_in_detect_class"
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

void postprocess_vgg16(nn_output *pout, char* result)
{
    float* densenet_ctc_buffer[1] = {NULL};
    densenet_ctc_buffer[0] = (float*)pout->out[0].buf;
    
    int i, max_index;
    int class_num = 10;
    float threshold = 0.5;
    float max_conf;
    
    max_conf = 0;
    max_index = 10;
    
    for (i = 0; i < class_num; ++i)
    {
    	if (densenet_ctc_buffer[0][i] > threshold && densenet_ctc_buffer[0][i] > max_conf)
    	{
    		max_conf = densenet_ctc_buffer[0][i];
    		max_index = i;
    	}
    }
    strncpy(result, names[max_index], 20);
}
