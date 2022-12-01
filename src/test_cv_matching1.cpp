#include <iostream>
#include "kb_cv_matching_akaze.h"
#include "kb_cv_RANSAC_calculate_transformation_matrix.h"
#include "kb_cv_paste.h"


//	対応点の可視化
void drawMatches(
	cv::Mat& mat1,
	cv::Mat& mat2,
	std::vector<cv::Point2f>& mp1i_out,
	std::vector<cv::Point2f>& mp2i_out,
	cv::Mat& matV12
)
{
	std::vector<cv::DMatch> v_dm_i;

	std::vector<cv::KeyPoint> v_kp1i, v_kp2i;
	int num_i = mp1i_out.size();
	for (int i = 0; i < num_i; i++) {
		cv::KeyPoint kp1, kp2;
		cv::DMatch dm;
		kp1.pt = mp1i_out[i];
		kp2.pt = mp2i_out[i];
		dm.queryIdx = i;
		dm.trainIdx = i;
		v_kp1i.push_back(kp1);
		v_kp2i.push_back(kp2);
		v_dm_i.push_back(dm);
	}

	cv::drawMatches(
		mat1, v_kp1i,
		mat2, v_kp2i,
		v_dm_i, matV12,
		cv::Scalar_<double>::all(-1),
		cv::Scalar_<double>::all(-1),
		std::vector<char>(),
		cv::DrawMatchesFlags::DEFAULT);
	//cv::imwrite(path_img_pair1, matV);


	//cv::namedWindow("matches", cv::WINDOW_NORMAL);
	//cv::imshow("matches", matV12);
	//cv::waitKeyEx(0);
}

int main(int argc, char* argv[])
{
	std::string path1 = argv[1];
	std::string path2 = argv[2];

	cv::Mat mat1 = cv::imread(path1);
	cv::Mat mat2 = cv::imread(path2);

	cv::Size sz1 = mat1.size();
	cv::Size sz2 = mat2.size();

	std::vector<cv::DMatch> v_dm;
	std::vector<cv::KeyPoint> v_kp1, v_kp2;

	//	対応点検索
	int rtn_wk = 0;
	kb::match_akaze(mat1, mat2, v_dm, v_kp1, v_kp2, 0.001, 1);

	cv::Mat matV12;
	{
		cv::drawMatches(
			mat1, v_kp1,
			mat2, v_kp2,
			v_dm, matV12,
			cv::Scalar_<double>::all(-1),
			cv::Scalar_<double>::all(-1),
			std::vector<char>(),
			cv::DrawMatchesFlags::DEFAULT);

		cv::namedWindow("12", cv::WINDOW_NORMAL);
		cv::imshow("12", matV12);

	}


	std::vector<cv::Point2f> mp1i, mp2i;
	int num_dm = v_dm.size();
	for (int i = 0; i < num_dm; i++) {
		int i1 = v_dm[i].queryIdx;
		int i2 = v_dm[i].trainIdx;
		mp1i.push_back(v_kp1[i1].pt);
		mp2i.push_back(v_kp2[i2].pt);
	}


	double threshold_dis1 = 10.0;
	int thres_max_valid_cnt = 40;
	int RANSAC_iter_num = 100;
	double det_min1 = 0.8;
	double det_max1 = 1.2;
	double threshold_ratio_delta_center = -1;
	double threshold_angle = -1;

	cv::Mat matAf;
	std::vector<cv::Point2f> mp1, mp2;

	kb::calculate_transformation_matrix_by_RANSAC(
		mp1i, 
		mp2i,
		sz1,
		matAf,
		mp1, 
		mp2,
		threshold_dis1,	
		thres_max_valid_cnt,
		RANSAC_iter_num,
		1,
		det_min1,
		det_max1,	
		1.0,				
		threshold_ratio_delta_center,
		threshold_angle,
		0				
	);

	std::cout << matAf << std::endl;

	cv::Mat matV12x;
	{
		drawMatches(mat1, mat2, mp1, mp2, matV12x);

		cv::namedWindow("12x", cv::WINDOW_NORMAL);
		cv::imshow("12x", matV12x);

	}



	cv::Mat matV1, matV2;
	kb::pasteTo(mat1, mat2, matAf, matV1, 0);
	kb::pasteTo(mat1, mat2, matAf, matV2, 1);

	cv::namedWindow("1", cv::WINDOW_NORMAL);
	cv::imshow("1", matV1);
	cv::namedWindow("2", cv::WINDOW_NORMAL);
	cv::imshow("2", matV2);

	cv::waitKeyEx(0);

	return 0;
}
