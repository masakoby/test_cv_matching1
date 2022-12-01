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
  
  
  
	cv::Size sz1 = mat1.size();
	cv::Size sz2 = mat2.size();

	std::vector<cv::DMatch> v_dm;
	std::vector<cv::KeyPoint> v_kp1, v_kp2;

	//	対応点検索
	int rtn_wk = 0;
	kb::match_akaze(mat1, mat2, v_dm, v_kp1, v_kp2, 0.001, 1);

	std::vector<cv::Point2f> mp1i, mp2i;
	{
		int num_dm = v_dm.size();
		for (int i = 0; i < num_dm; i++) {
			int i1 = v_dm[i].queryIdx;
			int i2 = v_dm[i].trainIdx;
			mp1.push_back(v_kp1[i1].pt);
			mp2.push_back(v_kp2[i2].pt);
		}

		int rtn=kb::calculate_transformation_matrix_by_RANSAC(
			mp1, mp2,
			sz1m,
			matAf,				//	計算結果の3x3行列
			mp1i,
			mp2i,
			threshold_dis1,		//	対応点一致するとみなす距離
			thres_max_valid_cnt,	//	対応点一致する最小数
			RANSAC_iter_num,		//	RANSAC試行回数
			1,//int numThread,				//	マルチスレッド
			det_min,				//	アフィン変換行列の2x2部分のdeterminant値の制限
			det_max,				//	アフィン変換行列の2x2部分のdeterminant値の制限	
			1.0,//double ratio_area,			//	画像の中心付近の対応点を優先的に使用するための係数
										//	　1.0 だとすべての点を均等に使用する
			threshold_ratio_delta_center,	//	画像中心の移動距離を閾値設定
			threshold_angle,
			0//int mode					//    0: affine transfromation
										// else: perspective transformation
		);

		std::cout << matAf << std::endl;

	cv::Mat matV1, matV2;
		kb::pasteTo(mat1, mat2, matAf, matV1, 0);
		kb::pasteTo(mat1, mat2, matAf, matV2, 1);

		cv::namedWindow("1", cv::WINDOW_NORMAL);
		cv::imshow("1", matV1);
		cv::namedWindow("2", cv::WINDOW_NORMAL);
		cv::imshow("2", matV2);
	}

  

  return 0;
}
