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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nn_sdk.h"
#include "nn_util.h"
#include "postprocess_util.h"
#include "nn_demo.h"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <getopt.h>

#define MODEL_WIDTH 280
#define MODEL_HEIGHT 32

struct option longopts[] = {
	{ "path",           required_argument,  NULL,   'p' },
	{ "model",          required_argument,  NULL,   'M' },
	{ "help",           no_argument,        NULL,   'H' },
	{ 0, 0, 0, 0 }
};

nn_input inData;

cv::Mat img;

aml_module_t modelType;

static int input_width,input_high,input_channel;

typedef enum _amlnn_detect_type_ {
	Accuracy_Detect_Yolo_V3 = 0
} amlnn_detect_type;

void* init_network_file(const char *mpath) {

	void *qcontext = NULL;

	aml_config config;
	memset(&config, 0, sizeof(aml_config));

	config.nbgType = NN_ADLA_FILE;
	config.path = mpath;
	config.modelType = ADLA_LOADABLE;
	config.typeSize = sizeof(aml_config);

	qcontext = aml_module_create(&config);
	if (qcontext == NULL) {
		printf("amlnn_init is fail\n");
		return NULL;
	}

	if (config.nbgType == NN_ADLA_MEMORY && config.pdata != NULL) {
		free((void*)config.pdata);
	}

	return qcontext;
}

int set_input(void *qcontext, const char *jpath) {

	int ret = 0;
	int input_size = 0;
	int hw = input_width*input_high;
	unsigned char *rawdata = NULL;
	cv::Mat temp_img(MODEL_WIDTH, MODEL_HEIGHT, CV_8UC1),normalized_img;
	img = cv::imread(jpath, 0);
	int width = img.cols;
	int height = img.rows;
	cv::resize(img, temp_img, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
	temp_img.convertTo(normalized_img, CV_32FC1, 1.0 / 255.0);

	rawdata = normalized_img.data;
	inData.input_type = BINARY_RAW_DATA;
	inData.input = rawdata;
	inData.input_index = 0;
	inData.size = input_width * input_high * input_channel * sizeof(float);
	ret = aml_module_input_set(qcontext, &inData);

	return ret;
}

int run_network(void *qcontext) {

	int ret = 0;
	nn_output *outdata = NULL;
	aml_output_config_t outconfig;
	memset(&outconfig, 0, sizeof(aml_output_config_t));

	outconfig.format = AML_OUTDATA_RAW;//AML_OUTDATA_RAW or AML_OUTDATA_FLOAT32
	outconfig.typeSize = sizeof(aml_output_config_t);
	
	outdata = (nn_output*)aml_module_output_get(qcontext, outconfig);
	if (outdata == NULL) {
		printf("aml_module_output_get error\n");
		return -1;
	}

	char result[35] = {0};
	int result_len = 0;
	
	postprocess_densenet_ctc(outdata, result, &result_len);
	
	printf("%d\n", result_len);
	printf("%s\n", result);

    return ret;
}

int destroy_network(void *qcontext) {

	int ret = aml_module_destroy(qcontext);
	return ret;
}

int main(int argc,char **argv)
{
	int c;
	int ret = 0;
	void *context = NULL;
	char *model_path = NULL;
	char *input_data = NULL;
	input_width = MODEL_WIDTH;
	input_high = MODEL_HEIGHT;
	input_channel = 1;

	while ((c = getopt_long(argc, argv, "p:m:H", longopts, NULL)) != -1) {
		switch (c) {
			case 'p':
				input_data = optarg;
				break;

			case 'm':
				model_path = optarg;
				break;

			default:
				printf("%s [-p picture path] [-m model path]  [-H]\n", argv[0]);
				exit(1);
		}
	}

	context = init_network_file(model_path);
	if (context == NULL) {
		printf("init_network fail.\n");
		return -1;
	}

	ret = set_input(context, input_data);

	if (ret != 0) {

		printf("set_input fail.\n");
		return -1;
	}

	ret = run_network(context);

	if (ret != 0) {
		printf("run_network fail.\n");
		return -1;
	}

	ret = destroy_network(context);

	if (ret != 0) {
		printf("destroy_network fail.\n");
		return -1;
	}

	return ret;
}
