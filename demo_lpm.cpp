#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include "lpm_matcher.h"

bool ReadMatchesFromTXT(const char* file_name, 
	std::vector<cv::Point2d>& query_pts, 
	std::vector<cv::Point2d>& refer_pts);

int main(int argc, char** argv) {

	//==================== Read the putative matches ====================//
	std::vector<cv::Point2d> query_pts, refer_pts;
    
	// The file path of the putative matches file
	std::string matches_file = std::string(SOURCE_DIR) + "/data/matches.txt";  
	ReadMatchesFromTXT(matches_file.c_str(), query_pts, refer_pts);

	//==================== LPM ====================//
	cv::TickMeter tm;
	tm.start();

	// Iteration 1
	LPM_Matcher lpm0(query_pts, refer_pts, 8, 0.8, 0.2);
	cv::Mat cost0;
	std::vector<bool> labels0;
	lpm0.Match(cost0, labels0);

	// Iteration 2
	LPM_Matcher lpm1(query_pts, refer_pts, 8, 0.5, 0.2, labels0);
	cv::Mat cost1;
	std::vector<bool> labels1;
	lpm1.Match(cost1, labels1);

	tm.stop();
	std::cout << "Cost time: " << tm.getTimeMilli() << " ms" << std::endl;
	
	//==================== Draw the matches ====================//
	
	// The file path of the query image
	std::string img0_file = std::string(SOURCE_DIR) + "/data/retina0.jpg";
	// The file path of the reference image
	std::string img1_file = std::string(SOURCE_DIR) + "/data/retina1.jpg";
	
	// Read the images
	cv::Mat img0 = cv::imread(img0_file);
	cv::Mat img1 = cv::imread(img1_file);
	
	cv::Mat concat_img;
	cv::hconcat(img0, img1, concat_img);

	for (int i = 0; i < static_cast<int>(query_pts.size()); ++i) {
		if (labels1[i]) { // Only show the inliers
			cv::line(concat_img, query_pts[i], refer_pts[i] +
				cv::Point2d(img0.cols, 0), CV_RGB(0, 255, 0), 2, 16);
		}
	}

	cv::imshow("matches", concat_img);
	cv::waitKey(0);

	return 0;
}

bool ReadMatchesFromTXT(const char* file_name, 
	std::vector<cv::Point2d>& query_pts, 
	std::vector<cv::Point2d>& refer_pts) {

	std::ifstream infile(file_name, std::ios::in);
	
	if (!infile.is_open()) {
		return false;
	}

	int num_points;
	infile >> num_points;
	query_pts.resize(num_points);
	refer_pts.resize(num_points);

	cv::Point2d pt;
	for (int i = 0; i < num_points; ++i) {
		infile >> pt.x;
		infile >> pt.y;
		query_pts[i] = pt;

		infile >> pt.x;
		infile >> pt.y;
		refer_pts[i] = pt;
	}

	return true;
}