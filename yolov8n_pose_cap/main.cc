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
#include "camera_util.h"
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
#include <sys/time.h>

#define MODEL_WIDTH 640
#define MODEL_HEIGHT 640

#define DEFAULT_WIDTH 1920
#define DEFAULT_HEIGHT 1080


int default_width = DEFAULT_WIDTH;
int default_height = DEFAULT_HEIGHT;

struct option longopts[] = {
	{ "device",         required_argument,  NULL,   'd' },
	{ "model",          required_argument,  NULL,   'm' },
	{ "width",          required_argument,  NULL,   'w' },
	{ "height",         required_argument,  NULL,   'h' },
	{ "type",           required_argument,  NULL,   't' },
	{ "help",           no_argument,        NULL,   'H' },
	{ 0, 0, 0, 0 }
};

int device = 0;

std::string type;

nn_input inData;

cv::Mat img;

aml_module_t modelType;

static int input_width,input_high,input_channel;

static const char *coco_names[] = {"person"};

int line_points[19][2] = {{0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 6}, {5, 7}, {5, 11}, {6, 8}, {6, 12}, {7, 9}, {8, 10}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}};
int point_colors[17] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2};
int line_colors[19] = {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2};
int pose_colors[3][3] = {{0, 255, 0}, {255, 0, 0}, {0, 0, 255}};

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

static cv::Scalar obj_id_to_color(int obj_id) {

	int const colors[6][3] = { { 1,0,1 }, { 0,0,1 }, { 0,1,1 }, { 0,1,0 }, { 1,1,0 }, { 1,0,0 } };
	int const offset = obj_id * 123457 % 6;
	int const color_scale = 150 + (obj_id * 123457) % 100;
	cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
	color *= color_scale;
	return color;
}

int run_network(void *qcontext) {

	int ret = 0, frames = 0;
	nn_output *outdata = NULL;
	struct timeval time_start, time_end;
	float total_time = 0;

	aml_output_config_t outconfig;
	memset(&outconfig, 0, sizeof(aml_output_config_t));

	outconfig.format = AML_OUTDATA_FLOAT32;//AML_OUTDATA_RAW or AML_OUTDATA_FLOAT32
	outconfig.typeSize = sizeof(aml_output_config_t);
	outconfig.order = AML_OUTPUT_ORDER_NCHW;

	pose_detect_out_t yolov3_detect_out;

	int classid = 0;
	float prob = 0;
	int left = 0, right = 0, top = 0, bot = 0;
	int point_x, point_y, point_x_0, point_y_0, point_x_1, point_y_1;

	cv::Point pt1;
	cv::Point pt2;

	int baseline;

	cv::namedWindow("Image Window");

	int input_size = 0;
	int hw = input_width*input_high;
	unsigned char *rawdata = NULL;
	cv::Mat temp_img(MODEL_WIDTH, MODEL_HEIGHT, CV_8UC1), normalized_img;

	inData.input_type = BINARY_RAW_DATA;
	inData.input_index = 0;
	inData.size = input_width * input_high * input_channel * sizeof(float);

	std::string device_number = std::to_string(device);
	if (type == "usb") {
		ret = load_usb_camera(device_number, default_width, default_height);
	}
	else if (type == "mipi") {
		//ret = load_mipi_camera(device_number, default_width, default_height);
		printf("V4L2 unsupports MIPI Camera now. If you want to use, you can use OpenCV call MIPI.\n");
		return -1;
	}
	else {
		printf("Unsupport camera type : %s\n", type.c_str());
		return -1;
	}

	while(1) {
		gettimeofday(&time_start, 0);
		if (type == "usb") {
			read_usb_frame(&img);
		}
		/*else if (type == "mipi") {
			read_mipi_frame(&img);
		}*/

		cv::resize(img, temp_img, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
		cv::cvtColor(temp_img, temp_img, cv::COLOR_RGB2BGR);
		temp_img.convertTo(normalized_img, CV_32FC3, 1.0 / 255.0);

		inData.input = normalized_img.data;

		ret = aml_module_input_set(qcontext, &inData);

		if (ret != 0) {
			printf("set_input fail.\n");
			return -1;
		}

		outdata = (nn_output*)aml_module_output_get(qcontext, outconfig);
		if (outdata == NULL) {
			printf("aml_module_output_get error\n");
			return -1;
		}

		postprocess_yolov3(outdata, &yolov3_detect_out);
		//printf("object_num:%d\n", yolov3_detect_out.detNum);

		for (int i =0;i < yolov3_detect_out.detNum;i++){
			classid = (int)yolov3_detect_out.pBox[i].objectClass;
			prob = yolov3_detect_out.pBox[i].score;
  	  
			left  = (yolov3_detect_out.pBox[i].x - yolov3_detect_out.pBox[i].w/2.) * img.cols;
			right = (yolov3_detect_out.pBox[i].x + yolov3_detect_out.pBox[i].w/2.) * img.cols;
			top   = (yolov3_detect_out.pBox[i].y - yolov3_detect_out.pBox[i].h/2.) * img.rows;
			bot   = (yolov3_detect_out.pBox[i].y + yolov3_detect_out.pBox[i].h/2.) * img.rows;
			//printf("class:%s,label_num:%d,prob:%f,left:%d,top:%d,right:%d,bot:%d\n",coco_names[classid], classid, prob, left, top, right, bot);

			pt1=cv::Point(left, top);
			pt2=cv::Point(right, bot);
			cv::Rect rect(left, top, right-left, bot-top);
			cv::rectangle(img, rect, obj_id_to_color(classid), 1, 8, 0);

			/*cv::Size text_size = cv::getTextSize(coco_names[classid], cv::FONT_HERSHEY_COMPLEX, 0.5 , 1, &baseline);
			cv::Rect rect1(left, top-20, text_size.width+10, 20);
			cv::rectangle(img, rect1, obj_id_to_color(classid), -1);
			cv::putText(img, coco_names[classid], cvPoint(left+5,top-5), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0,0,0), 1);*/
			
			for(int j = 0; j < 17; j++)
			{
				if (yolov3_detect_out.pose_points[i].point[j].prob >= 0.3)
				{
					point_x = yolov3_detect_out.pose_points[i].point[j].x * img.cols;
					point_y = yolov3_detect_out.pose_points[i].point[j].y * img.rows;
					cv::circle(img, cv::Point(point_x, point_y), 4, cv::Scalar(pose_colors[point_colors[j]][0], pose_colors[point_colors[j]][1], pose_colors[point_colors[j]][2]), -1);
				}
			}
			
			for(int j = 0; j < 19; j++)
			{
				if (yolov3_detect_out.pose_points[i].point[line_points[j][0]].prob >= 0.3 && yolov3_detect_out.pose_points[i].point[line_points[j][1]].prob >= 0.3)
				{
					point_x_0 = yolov3_detect_out.pose_points[i].point[line_points[j][0]].x * img.cols;
					point_y_0 = yolov3_detect_out.pose_points[i].point[line_points[j][0]].y * img.rows;
					point_x_1 = yolov3_detect_out.pose_points[i].point[line_points[j][1]].x * img.cols;
					point_y_1 = yolov3_detect_out.pose_points[i].point[line_points[j][1]].y * img.rows;
					cv::line(img, cv::Point(point_x_0, point_y_0), cv::Point(point_x_1, point_y_1), cv::Scalar(pose_colors[line_colors[j]][0], pose_colors[line_colors[j]][1], pose_colors[line_colors[j]][2]), 2);
				}
			}
		}
		cv::imshow("Image Window",img);
		cv::waitKey(1);
		
		gettimeofday(&time_end, 0);
		++frames;
		total_time += (float)((time_end.tv_sec - time_start.tv_sec) + (time_end.tv_usec - time_start.tv_usec) / 1000.0f / 1000.0f);
		if (total_time >= 1.0f) {
			int fps = (int)(frames / total_time);
			printf("Inference FPS: %i\n", fps);
			printf("Inference FPS: %f\n", total_time / frames);
			frames = 0;
			total_time = 0;
		}

	}
	if (type == "usb") {
		close_usb_camera();
	}
	/*else if (type == "mipi") {
		close_mipi_camera();
	}*/

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
	input_width = MODEL_WIDTH;
	input_high = MODEL_HEIGHT;
	input_channel = 3;

	while ((c = getopt_long(argc, argv, "d:m:w:h:t:H", longopts, NULL)) != -1) {
		switch (c) {
			case 'd':
				device = atoi(optarg);
				break;

			case 'w':
				default_width = atoi(optarg);
				break;

			case 'h':
				default_height = atoi(optarg);
				break;

			case 'm':
				model_path = optarg;
				break;
			
			case 't':
				type = optarg;
				break;

			default:
				printf("%s [-d device] [-m model path] [-w camera width] [-h camera height] [-t camera type]  [-H]\n", argv[0]);
				exit(1);
		}
	}

	if (type.empty()) {
		printf("Please choose camera type -t !!!\n");
		return -1;
	}
	
	context = init_network_file(model_path);
	if (context == NULL) {
		printf("init_network fail.\n");
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
