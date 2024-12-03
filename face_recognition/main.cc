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
#include "retinaface.h"
#include "facenet.h"
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

#define RETINAFACE_MODEL_WIDTH 640
#define RETINAFACE_MODEL_HEIGHT 640
#define FACENET_MODEL_WIDTH 160
#define FACENET_MODEL_HEIGHT 160

struct option longopts[] = {
	{ "path",              required_argument,  NULL,   'p' },
	{ "retinaface_model",  required_argument,  NULL,   'M' },
	{ "facenet_model",     required_argument,  NULL,   'm' },
	{ "help",              no_argument,        NULL,   'H' },
	{ 0, 0, 0, 0 }
};

int main(int argc,char **argv)
{
	int c;
	int ret = 0;
	char *input_data = NULL;
	cv::Mat img;
	face_detect_out_t retinaface_detect_out;
	float* facenet_result;
	float facenet_threshold = 0.6;
	
	float dst_landmark[5][2] = {{54.7065, 73.8519},
				    {105.0454, 73.5734},
				    {80.036, 102.4808},
				    {59.3561, 131.9507},
				    {89.6141, 131.7201}};
	/*float dst_landmark[5][2] = {{53, 53},
				    {107, 53},
				    {80, 80},
				    {53, 107},
				    {107, 107}};*/
	cv::Mat dst(5, 2, CV_32FC1, dst_landmark);
	memcpy(dst.data, dst_landmark, 2 * 5 * sizeof(float));
	
	void *retinaface_context = NULL;
	char *retinaface_model_path = NULL;
	int retinaface_input_width = RETINAFACE_MODEL_WIDTH;
	int retinaface_input_high = RETINAFACE_MODEL_HEIGHT;
	
	void *facenet_context = NULL;
	char *facenet_model_path = NULL;
	int facenet_input_width = FACENET_MODEL_WIDTH;
	int facenet_input_high = FACENET_MODEL_HEIGHT;
	
	int input_channel = 3;

	while ((c = getopt_long(argc, argv, "p:M:m:H", longopts, NULL)) != -1) {
		switch (c) {
			case 'p':
				input_data = optarg;
				break;

			case 'M':
				retinaface_model_path = optarg;
				break;
			
			case 'm':
				facenet_model_path = optarg;
				break;

			default:
				printf("%s [-p picture path] [-M retinaface model path] [-m facenet model path]  [-H]\n", argv[0]);
				exit(1);
		}
	}
	
	retinaface_context = init_retinaface_network_file(retinaface_model_path);
	facenet_context = init_facenet_network_file(facenet_model_path);
	
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
  		
		img = cv::imread(input_data);
		int x_padding = 0;
		int y_padding = 0;
		if (img.cols > img.rows)
		{
			y_padding = img.cols - img.rows;
		}
		else if (img.cols < img.rows)
		{
			x_padding = img.rows - img.cols;
		}
		
		ret = set_retinaface_input(retinaface_context, img, retinaface_input_width, retinaface_input_high, input_channel);
		ret = run_retinaface_network(retinaface_context, retinaface_detect_out);
	
		int classid = 0;
		float prob = 0;
		int left = 0, right = 0, top = 0, bot = 0;

		cv::Point pt1;
		cv::Point pt2;
	
		int baseline;
	
		for (int i =0;i < retinaface_detect_out.detNum;i++){
			float landmark[5][2] = {{retinaface_detect_out.point_1[i].x * (img.cols + x_padding), retinaface_detect_out.point_1[i].y * (img.rows + y_padding)},
						{retinaface_detect_out.point_2[i].x * (img.cols + x_padding), retinaface_detect_out.point_2[i].y * (img.rows + y_padding)},
						{retinaface_detect_out.point_3[i].x * (img.cols + x_padding), retinaface_detect_out.point_3[i].y * (img.rows + y_padding)},
						{retinaface_detect_out.point_4[i].x * (img.cols + x_padding), retinaface_detect_out.point_4[i].y * (img.rows + y_padding)},
						{retinaface_detect_out.point_5[i].x * (img.cols + x_padding), retinaface_detect_out.point_5[i].y * (img.rows + y_padding)}};
						
			cv::Mat src(5, 2, CV_32FC1, landmark);
			memcpy(src.data, landmark, 2 * 5 * sizeof(float));
		
			cv::Mat M = similarTransform(src, dst);
			cv::Mat warp;
			cv::warpPerspective(img, warp, M, cv::Size(FACENET_MODEL_WIDTH, FACENET_MODEL_HEIGHT));
		
			//cv::imwrite("./tmp.jpg", warp);
			
			ret = set_facenet_input(facenet_context, warp, facenet_input_width, facenet_input_high, input_channel);
			ret = run_facenet_network(facenet_context, &facenet_result);			
  			l2_normalize(facenet_result);
  			
  			float max_score = 0;
  			std::string name = "stanger";
  			while ((ptr = readdir(pDir)) != 0)
  			{
  				if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
  				{
  					std::cout << ptr->d_name << std::endl;
  					std::ifstream infile(face_lib + ptr->d_name);
  					std::string tmp;
  					float lib_feature[128];
  					float cos_similar;
  					
  					int i = 0;
  					while (getline(infile, tmp))
  					{
  						lib_feature[i] = atof(tmp.c_str());
  						i++;
  					}
  					infile.close();
  					
  					cos_similar = cos_similarity(facenet_result, lib_feature);
  					printf("%f\n", cos_similar);
  					if (cos_similar >= facenet_threshold && cos_similar > max_score)
  					{
  						max_score = cos_similar;
  						name = ((std::string)ptr->d_name).substr(0, ((std::string)ptr->d_name).find_last_of("."));
  					}
  				}
  			}
		
			classid = (int)retinaface_detect_out.pBox[i].objectClass;
			prob = retinaface_detect_out.pBox[i].score;
        
			left  = (retinaface_detect_out.pBox[i].x - retinaface_detect_out.pBox[i].w/2.) * (img.cols + x_padding);
			right = (retinaface_detect_out.pBox[i].x + retinaface_detect_out.pBox[i].w/2.) * (img.cols + x_padding);
			top   = (retinaface_detect_out.pBox[i].y - retinaface_detect_out.pBox[i].h/2.) * (img.rows + y_padding);
			bot   = (retinaface_detect_out.pBox[i].y + retinaface_detect_out.pBox[i].h/2.) * (img.rows + y_padding);
			if (left < 2) left = 2;
			if (right > img.cols-2) right = img.cols-2;
			if (top < 2) top = 2;
			if (bot > img.rows-2) bot = img.rows-2;
			printf("class:%s,label_num:%d,prob:%f,left:%d,top:%d,right:%d,bot:%d\n","face", classid, prob, left, top, right, bot);

			pt1=cv::Point(left, top);
			pt2=cv::Point(right, bot);
			cv::Rect rect(left, top, right-left, bot-top);
			cv::rectangle(img, rect, { 255,0,255 }, 1, 8, 0);

			if (top < 50) {
				top = 50;
				left +=10;
			}
			cv::Size text_size = cv::getTextSize(name, cv::FONT_HERSHEY_COMPLEX, 0.5 , 1, &baseline);
			cv::Rect rect1(left, top-20, text_size.width+10, 20);
			cv::rectangle(img, rect1, { 255,0,255 }, -1);
			cv::putText(img, name, cvPoint(left+5,top-5), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,0,0), 1);
		}
		closedir(pDir);
		cv::imwrite("output.bmp", img);
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
  			printf("Path doesn't Exist!\n");
  			return -1;
  		}
  		
  		while ((ptr = readdir(pDir)) != 0)
  		{
  			if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
  			{
  				img = cv::imread((path + ptr->d_name).c_str());
  				int x_padding = 0;
				int y_padding = 0;
				if (img.cols > img.rows)
				{
					y_padding = img.cols - img.rows;
				}
				else if (img.cols < img.rows)
				{
					x_padding = img.rows - img.cols;
				}
		
  				ret = set_retinaface_input(retinaface_context, img, retinaface_input_width, retinaface_input_high, input_channel);
  				
  				ret = run_retinaface_network(retinaface_context, retinaface_detect_out);
  				
  				if (retinaface_detect_out.detNum < 1)
  				{
  					std::cout << (path + ptr->d_name).c_str() << std::endl;
  					printf("Don't detect face!\n");
  					continue;
  				}
  				else if (retinaface_detect_out.detNum > 1)
  				{
  					std::cout << (path + ptr->d_name).c_str() << std::endl;
  					printf("Detect one more faces!\n");
  					continue;
  				}

  				float landmark[5][2] = {{retinaface_detect_out.point_1[0].x * (img.cols + x_padding), retinaface_detect_out.point_1[0].y * (img.rows + y_padding)},
							{retinaface_detect_out.point_2[0].x * (img.cols + x_padding), retinaface_detect_out.point_2[0].y * (img.rows + y_padding)},
							{retinaface_detect_out.point_3[0].x * (img.cols + x_padding), retinaface_detect_out.point_3[0].y * (img.rows + y_padding)},
							{retinaface_detect_out.point_4[0].x * (img.cols + x_padding), retinaface_detect_out.point_4[0].y * (img.rows + y_padding)},
							{retinaface_detect_out.point_5[0].x * (img.cols + x_padding), retinaface_detect_out.point_5[0].y * (img.rows + y_padding)}};

				cv::Mat src(5, 2, CV_32FC1, landmark);
				memcpy(src.data, landmark, 2 * 5 * sizeof(float));

				cv::Mat M = similarTransform(src, dst);
				cv::Mat warp;
				cv::warpPerspective(img, warp, M, cv::Size(FACENET_MODEL_WIDTH, FACENET_MODEL_HEIGHT));

				ret = set_facenet_input(facenet_context, warp, facenet_input_width, facenet_input_high, input_channel);
				ret = run_facenet_network(facenet_context, &facenet_result);
  				l2_normalize(facenet_result);

  				std::ofstream outfile(save_path + ((std::string)ptr->d_name).substr(0, ((std::string)ptr->d_name).find_last_of(".")) + ".dat", std::ofstream::out);
  				for (int i = 0; i < 128; ++i)
  				{
  					outfile << facenet_result[i] << "\n";
  				}
  				outfile.close();
  			}
  		}
  		closedir(pDir);
	}
	
	ret = destroy_retinaface_network(retinaface_context);
	ret = destroy_facenet_network(facenet_context);

	if (ret != 0) {
		printf("destroy_network fail.\n");
		return -1;
	}

	return ret;
}
