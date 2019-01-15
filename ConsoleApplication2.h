#pragma once

#include "resource.h"

//opencv
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/aruco/charuco.hpp"
#include "opencv2/aruco.hpp"
#include "algorithm"

//soft
#include "iostream"
#include "vector"
#include "stdio.h"
#include "stdlib.h"
#include "iostream"
#include "conio.h"
#include "Shlwapi.h"
#include "ctime"
#include "map"
//#include "iomanip"
#include "boost\format.hpp"


using namespace std;
using namespace cv;


int color_identify(Mat& in_img, Scalar target_color, int& contour_cnt, double threshold, double& rmse, 
	int index_ = 0, int mode_ = 0);

int distance_cal(Mat in_img, int threshold, double scale, int mode_);

vector<double> match_shape(Mat pattern, Mat& in_img, double thresh);

bool barcode_search(Mat& in_img, int& index);

int funcA(int n);

int index(int& op1, int& op2);

int add(int a, int b);