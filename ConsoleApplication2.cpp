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

	std::cout << "process start." << endl;
	
	vector<int> nums;
	for (int r = 0; r < 800; r++)
	{
		nums.push_back(rand()%10);
	}
	
	vector<int> answer;
	clock_t start_time;
	
	run_timer(true, start_time);
	
	for (int i = 0; i < nums.size(); i++)
	{
		bool sign = false;
		vector<int> check(nums.begin() + i, nums.end());
		check.insert(check.end(), nums.begin(), nums.begin() + i);
		for (int j = 0; j < check.size(); j++)
		{
			if (nums[i] < check[j])
			{
				answer.push_back(check[j]);
				sign = true;
				break;
			}
		}
		if (!sign)
		{
			answer.push_back(-1);
		}
	}

	double time_collapse = run_timer(false, start_time);
	std::cout << "Total time: "<< time_collapse * 1000.0 << " ms" << endl;

	

	

	///shape detection
	/*Mat pattern = imread("temp\\shape1.bmp");
	Mat in_img = imread("temp\\shapes2.bmp");
	vector<double> result = match_shape(pattern, in_img, 1.245);
	if (result.size() > 0)
	{
		cout << "Found: " << result.size() << endl;
		for (auto i : result)
		{
			cout << "result: " << i << endl;
		}
	}*/

	///barcode search
	/*int index_ = 0;
	string window_("image_window");
	namedWindow(window_, CV_WINDOW_AUTOSIZE);
	for (int i = 0; i < 6; i++)
	{
		ostringstream buffer;
		buffer << "temp\\barcode\\barcode" << index_ << ".jpg";
		string file_name = buffer.str();
		buffer.str("");
		Mat in_img = imread(file_name, 1);
		if (!in_img.data)
		{
			cout << "Invalid image" << endl;
			return 0;
		}
		bool sign = barcode_search(in_img, index_);
		cout << "Index: " << index_ << " result: " \
			<< boolalpha << sign << endl;
		buffer << "image_window" << index_;
		window_ = buffer.str();
		imshow(window_, in_img);
		cv::waitKey(100);
	}
	cv::waitKey(0);*/

	///AR marker detection
	//string message_;
	//BOOL save_sgn = TRUE;
	//Mat img1 = imread("tcam_ini1.bmp", 1);
	//Mat img2 = imread("scam_ini1.bmp", 1);
	//Mat img3 = imread("tcam_1.bmp", 1);//img_tcam_org_1
	//Mat img4 = imread("scam_1.bmp", 1);//scam_1
	//string index_ = "C920-118";//camera calibration parameter
	//int cnt_ = 50;
	//int camera_id = 1;
	//float mar = 40;
	//float squ = 50;
	//double valve_ = 0.02195;
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
	//posi_param.fs.open("coor1.xml", FileStorage::READ);
	//posi_param.fs["coor"] >> data_object;
	//posi_param.fs.release();
	////4
	//posi_param = posi_detect(camera_id, data_object, img3, img4, save_sgn);
	////5
	//posi_param = identify(img4, posi_param.r_points, valve_, save_sgn);
	//cout << "Inspection result: " << boolalpha << posi_param.iden_res << endl;
	//cout << "Inspect valve: " << posi_param.iden_valve << endl;

	///color
	//	//tape023: Scalar(140, 70, 36) tape14:Scalar(210, 110, 80) 
	//	//img05: Scalar(150, 85, 80) img06: Scalar(150, 95, 85) img07: Scalar(200, 100, 80) 35
	//	//top_1-2: Scalar(245, 100, 5)
	//	//Y Scalar(65, 125, 130)  Scalar(110, 195, 205)
	//	//M Scalar(25, 20, 80)
	//	//C Scalar(96, 50, 23)
	//	//K Scalar(3, 5, 3)
	//	//Y Scalar(45, 180, 190)
	//	//M Scalar(95, 55, 175) basler:Scalar(80, 45, 175)
	//	//C Scalar(190, 110, 15)
	//	//K Scalar(25, 40, 35)
	//	//Y Scalar(40, 220, 240)
	//	//M Scalar(90, 42, 189)
	//	//C Scalar(182, 114, 1)
	//	//Scalar(255, 193, 53)
	// //Scalar(21, 69, 187)
	//
	//string img_file = "temp\\E ring.jpg";//ymck\\test2.jpg
	//Mat img5 = imread(img_file);
	//if (!img5.data)
	//{
	//	cout << "No valid image takes in." << endl;
	//	cout << "process end." << endl;
	//	system("pause");
	//	return nRetCode;
	//}
	//int cnt = 0;
	//double RMSE = 0;
	//int code_ = color_identify(img5, Scalar(164, 166, 166), cnt, 10, RMSE, 17, 1);
	//if (code_ == -1)
	//{
	//	cout << "Input image invalid." << endl;
	//}
	//else if (code_ == -2)
	//{
	//	cout << "No such color detected or lesser than pre-set." << endl;
	//}
	//else
	//{
	//	cout << "Target Color Found: " << cnt << endl;
	//	cout << "RMSE: " << RMSE << endl;
	//}

	cout << "process end." << endl;
	cv::destroyAllWindows();
	system("pause");
    return nRetCode;
}

int color_identify(Mat& in_img, Scalar target_color, int& contour_cnt, double threshold, 
	double& rmse, int index_, int mode_)
{
	string file_name;
	ostringstream buffer;
	vector<Point2f> marker_;
	Mat res_;
	double tmp_rmse = 0;
	if (!in_img.data)
	{
		return -1;
	}
	Mat mask_ = Mat(in_img.rows, in_img.cols, CV_8UC1, Scalar::all(0));
	Mat element = getStructuringElement(MorphShapes::MORPH_RECT, Size(1, 1), Point(-1, -1));
	morphologyEx(in_img.clone(), res_, MorphTypes::MORPH_OPEN, 
		element, Point(-1, -1), 1);//MORPH_OPEN MORPH_CLOSE
	medianBlur(res_.clone(), res_, 1);
	for (int col = 0; col < res_.cols; col++)
	{
		for (int row = 0; row < res_.rows; row++)
		{
			Scalar color = res_.at<Vec3b>(row, col);
			rmse = norm(color[0] - target_color[0]) + norm(color[1] - target_color[1])
				+ norm(color[2] - target_color[2]);
			rmse = sqrt(rmse / 3.0);
			if (rmse < threshold)
			{
				mask_.at<uchar>(row, col) = 255;
				marker_.push_back(Point2f(col, row));
				if (tmp_rmse < rmse)
				{
					tmp_rmse = rmse;
				}
			}
		}
	}
	rmse = tmp_rmse;
	imwrite("temp\\mask.jpg", mask_);
//	distance_cal(mask_, 15, 0.03, 0);
	vector<vector<Point>> contours;
	findContours(mask_.clone(), contours, CV_RETR_EXTERNAL, 
		CV_CHAIN_APPROX_SIMPLE);//CV_RETR_LIST CV_RETR_EXTERNAL
	if (contours.size() < contour_cnt)
	{
		return -2;
	}
	contour_cnt = 0;
	vector<vector<Point>>::iterator itc = contours.begin();
	while (itc != contours.end())
	{
		double g_dConLength = arcLength(*itc, true);
		if (g_dConLength < 62 )//|| g_dConLength > 110
		{
			itc = contours.erase(itc);
		}
		else
		{
			contour_cnt++;
			Rect rec_ = boundingRect(*itc);
			//cout << g_dConLength << endl;
			//rectangle(in_img, rec_, Scalar(85, 114, 202), 4);
			++itc;
		}
	}
	if (mode_ > 0)
	{
		cv::drawContours(in_img, contours, -1, 
			Scalar(255- target_color[0], 255 - target_color[1], 255 - target_color[2]), 5, 8);
		//CV_FILLED Scalar(140, 183, 127)
		/*RotatedRect bound_ = minAreaRect(marker_);
		Point2f pts_[4];
		Point2f pts_reg[4];
		bound_.points(pts_);
		for (int p = 0; p < 4; p++)
		{
			line(in_img, pts_[p], pts_[(p + 1) % 4], Scalar(85, 114, 202), 4);
		}
		float box_w = norm(pts_[0] - pts_[3]);
		float box_h = norm(pts_[0] - pts_[1]);
		pts_reg[0] = Point2f(0, 0);
		pts_reg[1] = Point2f(box_w, 0);
		pts_reg[2] = Point2f(box_w, box_h);
		pts_reg[3] = Point2f(0, box_h);
		Mat P = getPerspectiveTransform(pts_, pts_reg);
		Mat perspect;
		warpPerspective(in_img.clone(), perspect, P, Size((int)box_w, (int)box_h));
		imwrite("temp\\perspective.jpg", perspect);*/
	}
	buffer << "temp\\res_" << index_ << ".jpg";
	file_name = buffer.str();
	buffer << "";
	imwrite(file_name, in_img);

	return 0;
}

int distance_cal(Mat in_img, int threshold, double scale, int mode_)
{
	Mat mask_ = in_img.clone();
	if (mask_.channels() > 1)
	{
		return 1;
	}
	///distance
#pragma region distance
	vector<double> dist_;
	Point* pts_ = new Point[100];
	double dist = 0.0f;
	double aver = 0.0f;
	double max_v = 0.0f;
	double min_v = 200.0f;
	Mat mask_2 = Mat(mask_.rows, mask_.cols, CV_8UC1, Scalar::all(0));
	if (mode_ == 0)//horizontal
	{
		for (int row = 0; row < mask_.rows; row++)
		{
			int ind = 0;
			for (int col = 1; col < mask_.cols - 1; col++)
			{
				if (mask_.at<uchar>(row, col) == 255 && //outer
					mask_.at<uchar>(row, col - 1) == 0 &&
					mask_.at<uchar>(row, col + 1) == 255)
				{
					pts_[ind] = Point(col, row);
					mask_2.at<uchar>(row, col) = 255;
					ind++;
				}
				if (mask_.at<uchar>(row, col) == 255 && //inner
					mask_.at<uchar>(row, col - 1) == 255 &&
					mask_.at<uchar>(row, col + 1) == 0)
				{
					pts_[ind] = Point(col, row);
					mask_2.at<uchar>(row, col) = 255;
					ind++;
				}
			}
			dist = abs(norm(pts_[0] - pts_[1]));
			if (dist > threshold)
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
		//		cout << "Dist" << row << "(pixels): " << dist << endl;
				cout << dist * scale << endl;//"Dist" << row << "(mm): " << 
			}
		}
	}
	else//vertical
	{
		for (int col = 0; col < mask_.cols; col++)
		{
			int ind = 0;
			for (int row = 1; row < mask_.rows - 1; row++)
			{
				if (mask_.at<uchar>(row, col) == 255 && //outer
					mask_.at<uchar>(row - 1, col) == 0 &&
					mask_.at<uchar>(row + 1, col) == 255)
				{
					pts_[ind] = Point(col, row);
					ind++;
				}
				if (mask_.at<uchar>(row, col) == 255 && //inner
					mask_.at<uchar>(row - 1, col) == 255 &&
					mask_.at<uchar>(row + 1, col) == 0)
				{
					pts_[ind] = Point(col, row);
					ind++;
				}
				mask_2.at<uchar>(row, col) = 255;
			}
			dist = abs(norm(pts_[0] - pts_[1]));
			if (dist > threshold)
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
			//	cout << "Dist" << col << "(pixels): " << dist << endl;
			//	cout << "Dist" << col << "(mm): " << dist * scale << endl;
			}
		}
	}
	aver /= dist_.size();
	cout << "averDist(pixels): " << aver << endl;
	cout << "averDist(mm): " << aver * scale << endl;
	cout << "MaxDist(pixels): " << max_v << endl;
	cout << "MaxDist(mm): " << max_v * scale << endl;
	cout << "MinDist(pixels): " << min_v << endl;
	cout << "MinDist(mm): " << min_v * scale << endl;
	pts_ = NULL;
	delete[] pts_;
	imwrite("temp\\mask2.bmp", mask_2);
#pragma endregion
	return 0;
}

vector<double> match_shape(Mat pattern, Mat& in_img, double thresh)
{
	vector<double> threshold;
	Mat original_img = in_img.clone();
	if (pattern.channels() > 1)
	{
		cvtColor(pattern.clone(), pattern, CV_BGR2GRAY);
		cv::adaptiveThreshold(pattern.clone(), pattern, 255, 
			ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 21, 15);
	}
	if (in_img.channels() > 1)
	{
		cvtColor(in_img.clone(), in_img, CV_BGR2GRAY);
		cv::adaptiveThreshold(in_img.clone(), in_img, 255, 
			ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 21, 15);
	}

	vector<vector<Point>> in_pattern;
	findContours(pattern, in_pattern, CV_RETR_LIST,
		CV_CHAIN_APPROX_NONE);//CV_RETR_EXTERNAL CV_RETR_LIST
	
	vector<vector<Point>> in_contour;
	findContours(in_img, in_contour, CV_RETR_LIST,
		CV_CHAIN_APPROX_NONE);//CV_RETR_EXTERNAL
	vector<vector<Point>>::iterator itc = in_contour.begin();
	while (itc != in_contour.end())
	{
		double g_dConLength = arcLength(*itc, true);
		if (g_dConLength < 50 || g_dConLength > 1000)
		{
			itc = in_contour.erase(itc);
		}
		else
		{
			double threshold_ = cv::matchShapes(in_pattern[0], *itc, 
				CV_CONTOURS_MATCH_I2, 0);
			if (threshold_ < thresh)
			{
				threshold.push_back(threshold_);
				cv::drawContours(original_img, in_contour, itc- in_contour.begin(),
					Scalar(rand() % 255, rand() % 255, rand() % 255), 2, 8); 
				//distance(in_contour.begin(), itc)
			}
			++itc;
		}
	}
	if (threshold.size() > 0)
	{
		imwrite("result.bmp", original_img);
	}
	return threshold;
}

bool barcode_search(Mat& in_img, int& index)
{
	bool found = true;
	Mat original_img = in_img.clone();
	if (in_img.channels() > 1)
	{
		cvtColor(in_img.clone(), in_img, CV_BGR2GRAY);
	}
	Mat temp;
	cv::morphologyEx(in_img, temp, MorphTypes::MORPH_GRADIENT, 
		getStructuringElement(MorphShapes::MORPH_RECT, Size(7, 1)));
	medianBlur(temp.clone(), temp, 21);
	threshold(temp.clone(), temp, 66, 255, THRESH_BINARY);
	cv::morphologyEx(temp.clone(), temp, MorphTypes::MORPH_OPEN,
		getStructuringElement(MorphShapes::MORPH_RECT, Size(3, 3)), Point(-1, -1), 4);
	cv::morphologyEx(temp.clone(), temp, MorphTypes::MORPH_CLOSE,
		getStructuringElement(MorphShapes::MORPH_RECT, Size(3, 3)), Point(-1, -1), 4);
	vector<vector<Point>> in_contour;
	findContours(temp, in_contour, CV_RETR_EXTERNAL,
		CV_CHAIN_APPROX_SIMPLE);//CV_RETR_EXTERNAL
	vector<vector<Point>>::iterator itc = in_contour.begin();
	while (itc != in_contour.end())
	{
		double g_dConLength = arcLength(*itc, true);
		if (g_dConLength < 450 || g_dConLength > 1300)
		{
			itc = in_contour.erase(itc);
		}
		else
		{
			Rect rectangle_ = boundingRect(*itc);
			rectangle(original_img, rectangle_,
				Scalar(rand() % 255, rand() % 255, rand() % 255), 3);
			++itc;
		}
	}
	if (in_contour.size() == 0)
	{
		found = false;
		return found;
	}
	ostringstream buffer;
	buffer << "temp\\barcode\\barcode_fnd_result" << index << ".bmp";
	string file_name = buffer.str();
	buffer << "";
	imwrite(file_name, original_img);
	index++;
	in_img = original_img;
	return found;
}

int funcA(int n)
{
	if (n>1)
	{
		int num = n + funcA(n - 1);
		cout << "part result: " << num << endl;
		return num;
	}
	return n;
}

int index(int& op1, int& op2)//C42
{
	int res = 1;
	for (int i = 1; i < op1; i++)
	{
		op2 *= (op2 - 1);
	}
	for (int j = op1; j > 1; j--)
	{
		op1 *= (op1 - 1);
	}
	res = op2 / op1;
	return res;
}

int add(int a, int b)
{
	return (b == 0) ? a : add(a ^ b, (a & b) << 1);
}

double run_timer(bool switch_, clock_t& start_time)
{
	double time_collapse = 0.0f;
	if (switch_)//on
	{
		start_time = clock();
	}
	else
	{
		clock_t end_time = clock();
		time_collapse = (double)(end_time - start_time) / CLOCKS_PER_SEC;
	}
	return time_collapse;
}