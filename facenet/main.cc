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
#include <dirent.h>
#include <iostream>
#include <fstream>

#define MODEL_WIDTH 160
#define MODEL_HEIGHT 160

struct option longopts[] = {
	{ "path",           required_argument,  NULL,   'p' },
	{ "model",          required_argument,  NULL,   'M' },
	{ "help",           no_argument,        NULL,   'H' },
	{ 0, 0, 0, 0 }
};

nn_input inData;

cv::Mat orig_img;
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
	cv::Mat temp_img(MODEL_WIDTH, MODEL_HEIGHT, CV_8UC3),normalized_img;
	orig_img = cv::imread(jpath);
	int width = img.cols;
	int height = img.rows;
	cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
	if (orig_img.cols >= orig_img.rows)
  	{
  		int y_padding = orig_img.cols - orig_img.rows;
  		cv::copyMakeBorder(img, img, 0, y_padding, 0, 0, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  		cv::resize(img, temp_img, cv::Size(160, 160));
  	}
  	else
  	{
  		int x_padding = orig_img.rows - orig_img.cols;
  		cv::copyMakeBorder(img, img, 0, 0, 0, x_padding, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  		cv::resize(img, temp_img, cv::Size(160, 160));
  	}
	temp_img.convertTo(normalized_img, CV_32FC3, 1.0 / 255.0);

	rawdata = normalized_img.data;
	inData.input_type = BINARY_RAW_DATA;
	inData.input = rawdata;
	inData.input_index = 0;
	inData.size = input_width * input_high * input_channel * sizeof(float);
	ret = aml_module_input_set(qcontext, &inData);

	return ret;
}

int run_network(void *qcontext, float** result) {

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
	
	result[0] = (float*)outdata->out[0].buf;

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
	input_channel = 3;

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
	
	float* result;
	
	if (!(atoi(input_data) == 1))
	{
		std::string face_lib = "../face_feature_lib/";
  		DIR *pDir;
  		struct dirent *ptr;
  		if (!(pDir = opendir(face_lib.c_str())))
  		{
  			printf("Feature library doesn't Exist!\n");
  			return -1;
  		}
  		
  		ret = set_input(context, input_data);
		if (ret != 0) {
			printf("set_input fail.\n");
			return -1;
		}
		
		ret = run_network(context, &result);
  		if (ret != 0) {
			printf("run_network fail.\n");
			return -1;
		}
  		
  		l2_normalize(result);
  		
  		while ((ptr = readdir(pDir)) != 0)
  		{
  			if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
  			{
  				std::ifstream infile(face_lib + ptr->d_name);
  				std::cout << "\n" << ptr->d_name << std::endl;
  				std::string tmp;
  				float lib_feature[128];
  				float tmp2, tmp3;
  				
  				int i = 0;
  				while (getline(infile, tmp))
  				{
  					lib_feature[i] = atof(tmp.c_str());
  					i++;
  				}
  				infile.close();
  				
  				tmp2 = compare_eu_distance(result, lib_feature);
  				tmp3 = cos_similarity(result, lib_feature);
  				printf("eu_distance:%f\n", tmp2);
  				printf("cos_similarity:%f\n", tmp3);
  			}
  		}
	}
	else
	{
		std::string save_path = "../face_feature_lib/";
		std::string path = "../data/img/";
		DIR *pDir;
  		struct dirent *ptr;
  		if (!(pDir = opendir(save_path.c_str())))
  		{
  			system(("mkdir " + save_path).c_str());
  		}
  		if (!(pDir = opendir(path.c_str())))
  		{
  			printf("Face path doesn't Exist!\n");
  			return -1;
  		}
  		while ((ptr = readdir(pDir)) != 0)
  		{
  			if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
  			{
  				ret = set_input(context, (path + ptr->d_name).c_str());
				if (ret != 0) {
					printf("set_input fail.\n");
					return -1;
				}
				
  				ret = run_network(context, &result);
  				if (ret != 0) {
					printf("run_network fail.\n");
					return -1;
				}
  				
  				l2_normalize(result);
  				
  				std::ofstream outfile(save_path + ((std::string)ptr->d_name).substr(0, ((std::string)ptr->d_name).find_last_of(".")) + ".dat", std::ofstream::out);
  				for (int i = 0; i < 128; ++i)
  				{
  					outfile << result[i] << "\n";
  				}
  				outfile.close();
  			}
  		}
	}

	ret = destroy_network(context);

	if (ret != 0) {
		printf("destroy_network fail.\n");
		return -1;
	}

	return ret;
}
