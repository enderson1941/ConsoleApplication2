#pragma once
#include "stdafx.h"
#include "opencv2/aruco/charuco.hpp"
#include "aruco.hpp"
#include "Shlwapi.h"

//namespace
using namespace std;
using namespace cv;
using namespace aruco;

int lp_var = 0;

//struct
struct  posi_paramter
{
	int							marker_cnt = 20;
	bool						op_sgn = TRUE;
	bool						init_sgn = TRUE;
	bool						iden_res;
	double					set_valve = 0.65;
	double					iden_valve;
	Mat						R_t;
	Mat						tmp_;
	Mat						res_img;
	Mat						iden_img;
	Point3f				marker_coor;
	Point2f					r_points[4];
	FileStorage			fs;
	vector<Vec3d>		rvecs, tvecs;
	Vec3d					_rvecs, _tvecs;
	vector<Point3f>	dted_marker_coor;
	CString					tmp_msg;
	CString					wrn_msg[4];//
};
posi_paramter posi_param;

struct calib_paramter
{
	int										calibrationFlags = 0;
	int										frame_id = 0;//
	int										frame_cnt = 50;//
	int										dictionaryId = 8;
	float									aspectRatio = 1;
	float									squareLength = 40;
	float									markerLength = 30;
	bool									refindStrategy = false;
	bool									calibed = true;
	string								camera_index;//
	CvSize								calib_board = CvSize(5, 7);

	// collect data from each frame
	vector<vector<int>>			allIds;
	vector<Mat>						allImgs;
	Mat									calib_img;
	Mat									res_img;
	VideoCapture						cam_cap;
	Size									imgSize;
	Mat									cameraMatrix, distCoeffs;
	Mat									currentCharucoCorners, currentCharucoIds;
	vector<int>						markerIds;
	vector<Point2f>				markerCenter;
	vector<vector<Point2f>>	markerCorners, rejectedCandidates;
	vector<vector<vector<Point2f>>>		allCorners;
	Dictionary							calib_dict;
	DetectorParameters			detectorParams;
	CharucoBoard					charucoboard;

	//estimate
	Mat									rvecs, tvecs;
	Mat									Hw, Vw;
	Mat									tmp;

	// system
	BOOL								op_sgn = FALSE;//
	BOOL								init_sgn = FALSE;//
	BOOL								calib_sgn = FALSE;//
	string								outputFile = "default_camera.xml";//
	CString								wrn_msg[4];//
};
calib_paramter calib_param;

//function
extern "C" __declspec(dllexport) calib_paramter camera_calib_init(string& cam_index, Size& calib_board, 
	float& marker_length, float& square_length, int frame_cnt = 50, int marker_cnt = 20);

extern "C" __declspec(dllexport) calib_paramter camera_calib(Mat calib_frame);

extern "C" __declspec(dllexport) bool detect_markers_(Mat in_img, int mode_ = 0, int cam_id = 0);

extern "C" __declspec(dllexport) BOOL readDetectorParameters(string detect_param, DetectorParameters &params);

extern "C" __declspec(dllexport) BOOL calib_calculate();

extern "C" __declspec(dllexport) BOOL saveCameraParams(const string &filename, Size imageSize, 
	float aspectRatio, int flags, const Mat &cameraMatrix, const Mat &distCoeffs, double totalAvgErr);

extern "C" __declspec(dllexport) posi_paramter posi_detect_init(int cam_id, Mat top_img, Mat scan_img);

extern "C" __declspec(dllexport) posi_paramter posi_detect(int cam_id, Mat _data_, Mat in_img, 
	Mat inspect_img = Mat(1280, 720, CV_32FC1, Scalar(255)), BOOL save_op = FALSE);

extern "C" __declspec(dllexport) posi_paramter identify(Mat in_image, const Point2f* coordinates,  double& valve, bool sav_op);
//