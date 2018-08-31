// ConsoleApplication2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ConsoleApplication2.h"
#include "algorithm_lib.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// The one and only application object

CWinApp theApp;

using namespace std;

int main()
{
    int nRetCode = 0;

    HMODULE hModule = ::GetModuleHandle(nullptr);

    if (hModule != nullptr)
    {
        // initialize MFC and print and error on failure
        if (!AfxWinInit(hModule, nullptr, ::GetCommandLine(), 0))
        {
            // TODO: change error code to suit your needs
            wprintf(L"Fatal Error: MFC initialization failed\n");
            nRetCode = 1;
        }
        else
        {
            // TODO: code your application's behavior here.
        }
    }
    else
    {
        // TODO: change error code to suit your needs
        wprintf(L"Fatal Error: GetModuleHandle failed\n");
        nRetCode = 1;
    }

	string message_;

	BOOL save_sgn = TRUE;
	Mat img1 = imread("tcam_ini1.bmp", 1);
	Mat img2 = imread("scam_ini1.bmp", 1);
	Mat img3 = imread("img_tcam_org_1.bmp", 1);//img_tcam_org_1
	Mat img4 = imread("img_tcam_org_1.bmp", 1);//scam_1

	string index_ = "C920-118";//camera calibration parameter
	int cnt_ = 50;
	int camera_id = 0;
	float mar = 40;
	float squ = 50;
	double valve_ = 0.02195;

	cout << "process start." << endl;
	//Mat data_object;
	////1
	//calib_param = camera_calib_init(index_, cv::Size(7, 5), mar, squ);
	////2
	//posi_param = posi_detect_init(camera_id, img1, img2);
	//if (!posi_param.op_sgn)
	//{
	//	message_ = CW2A(posi_param.wrn_msg[0].GetString());
	//	cout << message_ << endl;
	//	nRetCode = 1;
	//	cout << "process end." << endl;
	//	system("pause");
	//	return nRetCode;
	//}
	////3
	//posi_param.fs.open("coor.xml", FileStorage::READ);
	//posi_param.fs["coor"] >> data_object;
	//posi_param.fs.release();
	////4
	//posi_param = posi_detect(camera_id, data_object, img3, img4, save_sgn);
	////5
	//posi_param = identify(img4, posi_param.r_points, valve_, save_sgn);
	//cout << "Inspection result: " << boolalpha << posi_param.iden_res << endl;
	//cout << "Inspect valve: " << posi_param.iden_valve << endl;
	
	string img_file = "temp//img00.jpg";
	Mat img5 = imread(img_file);
	int cnt = 2;
	int code_ = color_identify(img5, Scalar(140, 70, 36), cnt, 35, 0);// Scalar(140, 70, 36) Scalar(150, 85, 80) 35
	if (code_<0)
	{
		cout << "No such color detected or lesser than pre-set." << endl;
	}
	else
	{
		cout << "Color Found: " << cnt << endl;
	}

	cout << "process end." << endl;
	system("pause");
    return nRetCode;
}

int color_identify(Mat& in_img, Scalar target_color, int& contour_cnt, double threshold, int index_)
{
	string file_name;
	ostringstream buffer;
	vector<Point2f> marker_;
	Mat res_;
	Mat mask_ = Mat(in_img.rows, in_img.cols, CV_8UC1, Scalar::all(0));
	Mat element = getStructuringElement(MorphShapes::MORPH_RECT, Size(3, 3), Point(-1, -1));
	morphologyEx(in_img, res_, MorphTypes::MORPH_OPEN, element, Point(-1, -1), 4);
	//imwrite("temp//morph.jpg", res_);
	for (int col = 0; col < res_.cols; col++)
	{
		for (int row = 0; row < res_.rows; row++)
		{
			Scalar color = res_.at<Vec3b>(row, col);
			double tmp = sqrt(norm(color[0] - target_color[0]) + norm(color[1] - target_color[1])+ norm(color[2] - target_color[2]));
			if (tmp < threshold)
			{
				mask_.at<uchar>(row, col) = 255;
				marker_.push_back(Point2f(col, row));
			}
		}
	}
	//Canny(mask_.clone(), mask_, 17, 125);
	//normalize(mask_, mask_, 0, 255, NORM_MINMAX);
	//imwrite("temp//mask.jpg", mask_);

	/*Mat sobel_x, sobel_y, angle;
	Sobel(mask_.clone(), sobel_x, CV_64FC1, 1, 0);
	Sobel(mask_.clone(), sobel_y, CV_64FC1, 0, 1);
	phase(sobel_x, sobel_y, angle, true); 
	normalize(angle, angle, 0, 255, NORM_MINMAX); 
	angle.convertTo(angle, CV_8UC1);
	imwrite("temp//angle.jpg", angle);
	bitwise_or(sobel_x, sobel_y, angle);
	imwrite("temp//combine.jpg", angle);*/

	/*RotatedRect bound_ = minAreaRect(marker_);
	Point2f pts_[4];
	bound_.points(pts_);
	for (int p = 0; p < 4; p++)
	{
		line(in_img, pts_[p], pts_[(p + 1) % 4], Scalar(0, 255, 0), 2);
	}*/

	vector<vector<Point>> contours;
	findContours(mask_.clone(), contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);//CV_RETR_LIST
	if (contours.size() < contour_cnt)
	{
		return -1;
	}
	contour_cnt = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		Rect rec_ = boundingRect(contours[i]);
		if (rec_.area() > 30)//30
		{
			rectangle(in_img, rec_, Scalar(0, 255, 0), 2);
			contour_cnt++;
		}
	}
	buffer << "temp//res_" << index_ << ".jpg";
	file_name = buffer.str();
	buffer << "";
	imwrite(file_name, in_img);
	return 0;
}
