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

	string img_file = "temp\\top_2.png";
	Mat img5 = imread(img_file);
	int cnt = 0;
	int code_ = color_identify(img5, Scalar(245, 100, 5), cnt, 65, 9, 1);
	// tape023: Scalar(140, 70, 36) tape14:Scalar(210, 110, 80) 
	// img05: Scalar(150, 85, 80) img06: Scalar(150, 95, 85) img07: Scalar(200, 100, 80) 35
	// top_1-2: Scalar(245, 100, 5)
	if (code_ == -1)
	{
		cout << "Input image invalid." << endl;
	}
	else if (code_ == -2)
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

int color_identify(Mat& in_img, Scalar target_color, int& contour_cnt, double threshold, int index_, int mode_)
{
	string file_name;
	ostringstream buffer;
	vector<Point2f> marker_;
	vector<double> dist_;
	Mat res_;
	if (!in_img.data)
	{
		return -1;
	}
	Mat mask_ = Mat(in_img.rows, in_img.cols, CV_8UC1, Scalar::all(0));
	Mat element = getStructuringElement(MorphShapes::MORPH_RECT, Size(3, 3), Point(-1, -1));
	/*morphologyEx(in_img, res_, MorphTypes::MORPH_TOPHAT, element, Point(-1, -1), 5);
	imwrite("temp\\TOPHAT.jpg", res_);
	res_ = in_img - res_;
	imwrite("temp\\minus.jpg", res_);*/
	morphologyEx(in_img.clone(), res_, MorphTypes::MORPH_OPEN, element, Point(-1, -1), 3);//MORPH_OPEN MORPH_CLOSE
	medianBlur(res_.clone(), res_, 1);
	for (int col = 0; col < res_.cols; col++)
	{
		for (int row = 0; row < res_.rows; row++)
		{
			Scalar color = res_.at<Vec3b>(row, col);
			double tmp = sqrt(norm(color[0] - target_color[0]) + 
				norm(color[1] - target_color[1]) + norm(color[2] - target_color[2]));
			if (tmp < threshold)
			{
				mask_.at<uchar>(row, col) = 255;
				marker_.push_back(Point2f(col, row));
			}
		}
	}
	imwrite("temp\\mask.jpg", mask_);
	Point* pts_  = new Point[10];
	double dist = 0.0f;
	double aver = 0.0f;
	double max_v = 0.0f;
	double min_v = 200.0f;
//	Mat mask_2 = Mat(mask_.rows, mask_.cols, mask_.type(), Scalar::all(0));
	for (int row = 0; row < mask_.rows; row++)
	{
		int ind = 0;
		for (int col = 0; col < mask_.cols; col++)
		{
			if (mask_.at<uchar>(row, col) == 255 && //outer
				mask_.at<uchar>(row , col - 1 ) == 0 && 
				mask_.at<uchar>(row , col + 1) == 255  )
			{
			//	mask_2.at<uchar>(row, col) = 255;
				pts_[ind] = Point(col, row);
				ind++;
			}
			if (mask_.at<uchar>(row, col) == 255 && //inner
				mask_.at<uchar>(row, col - 1) == 255 &&
				mask_.at<uchar>(row, col + 1) == 0)
			{
			//	mask_2.at<uchar>(row, col) = 255;
				pts_[ind] = Point(col, row);
				ind++;
			}
		}
		dist = abs(norm(pts_[0] - pts_[1]));
		if (dist > 15)
		{
			dist_.push_back(dist);
			aver += dist;
			if (dist > max_v)
			{
				max_v = dist;
			}
			if (dist < min_v)
			{
				min_v = dist;
			}
			cout << "Dist" << row << "(pixels): " << dist << endl;
		}
	}
	aver /= dist_.size();
	cout << "averDist(pixels): " << aver << endl;
	cout << "MaxDist(pixels): " << max_v << endl;
	cout << "MinDist(pixels): " << min_v << endl;
//	imwrite("temp\\mask2.jpg", mask_2);

	///others
	//vector<vector<Point>> contours;
	//findContours(mask_.clone(), contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);//CV_RETR_LIST CV_RETR_EXTERNAL
	//if (contours.size() < contour_cnt)
	//{
	//	return -2;
	//}
	//contour_cnt = 0;
	//vector<vector<Point>>::iterator itc = contours.begin();
	//while (itc != contours.end())
	//{
	//	if (itc->size() < 130)
	//	{
	//		itc = contours.erase(itc);
	//	}
	//	else
	//	{
	//		contour_cnt++;
	//		Rect rec_ = boundingRect(*itc);
	////		rectangle(in_img, rec_, Scalar(85, 114, 202), 4);
	//		++itc;
	//	}
	//}
	//
	//if (mode_ > 0)
	//{
	//	cv::drawContours(in_img, contours, -1, Scalar(140, 183, 127), 4, 8);//CV_FILLED
	//	RotatedRect bound_ = minAreaRect(marker_);
	//	Point2f pts_[4];
	//	Point2f pts_reg[4];
	//	bound_.points(pts_);
	//	for (int p = 0; p < 4; p++)
	//	{
	//		line(in_img, pts_[p], pts_[(p + 1) % 4], Scalar(85, 114, 202), 4);
	//	}
	//	float box_w = norm(pts_[0] - pts_[3]);
	//	float box_h = norm(pts_[0] - pts_[1]);
	//	pts_reg[0] = Point2f(0, 0);
	//	pts_reg[1] = Point2f(box_w, 0);
	//	pts_reg[2] = Point2f(box_w, box_h);
	//	pts_reg[3] = Point2f(0, box_h);
	//	Mat P = getPerspectiveTransform(pts_, pts_reg);
	//	Mat perspect;
	//	warpPerspective(in_img.clone(), perspect, P, Size((int)box_w, (int)box_h));
	//	imwrite("temp\\perspective.jpg", perspect);
	//}
	//
	//buffer << "temp\\res_" << index_ << ".jpg";
	//file_name = buffer.str();
	//buffer << "";
	//imwrite(file_name, in_img);

	pts_ = NULL;
	delete[] pts_;
	return 0;
}
