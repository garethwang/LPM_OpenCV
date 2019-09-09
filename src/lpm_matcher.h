/****************************************************************************************************
 * @file lpm_matcher.h
 * @author Gareth Wang <gareth.wang@hotmail.com>
 * @brief The C++ implementation of the LPM algorithm.
 * @details Read the paper "Locality Preserving Matching." written by Jiayi Ma, et al. for details.
 * @date 2019-09-04
 *
 * @copyright Copyright (c) 2019
 *
*****************************************************************************************************/
#ifndef _LPM_MATCHER_H_
#define _LPM_MATCHER_H_
#include <opencv2/opencv.hpp>

/**
 * @brief The class for locality preserving matching. 
 */
class LPM_Matcher
{
public:
	/**
	 * @brief  Constructor.
	 *
	 * @param  query_points [in] The vector of \f$ N\f$ points from the query image.
	 * @param  refer_points [in] The vector of \f$ N\f$ points from the reference image.
	 * @param  knn [in] The number of nearest neighbors.
	 * @param  lambda [in] \f$ \lambda\f$.
	 * @param  tau [in] \f$ \tau\f$.
	 * @param  labels [in] The vector of \f$ N\f$ elements, every element of which is set to 0 for outliers and to 1 for the other points.
	 */
	LPM_Matcher(const std::vector<cv::Point2d>& query_points, 
		const std::vector<cv::Point2d>& refer_points, 
		const int knn = 8, const double lambda = 0.9, const double tau = 0.2,
		const std::vector<bool>& labels = std::vector<bool>());
	
	~LPM_Matcher();

	/**
	 * @brief  Performs the locality preserving matching. 
	 *
	 * @return void 
	 * @param  cost [out] The costs of the putative matches.
	 * @param  labels [out] The binary vector that represents the match correctness of the correspondences.
	 */
	void Match(cv::Mat& cost, std::vector<bool>& labels);

private:
	/**
	 * @brief  Converts the putative matches into displacement vectors and finds the k-nearest neighbor of the feature points.
	 *
	 * @return void 
	 * @param  labels [in] The binary vector that represents the match correctness of the correspondences.
	 */
	void Initialize(const std::vector<bool>& labels);

	/**
	 * @brief  Finds common elements in the two neighborhoods of the feature points from the query image and the reference image.
	 *
	 * @return std::vector<std::vector<int>> The indices of common elements in the two neighborhoods.
	 * @param  query_knn [in] The K-NN of the feature points from the query image.
	 * @param  refer_knn [in] The K-NN of the feature points from the reference image.
	 * 
	 */
	std::vector<std::vector<int> > FindNeighborsIntersection(
		const cv::Mat& query_knn, const cv::Mat& refer_knn) const;

	/**
	 * @brief  Computes the cost of each putative correspondence using a fixed \f$ K\f$.
	 *
	 * @return cv::Mat The costs of the putative matches.
	 * @param  query_knn [in] The K-NN of the feature points from the query image.
	 * @param  refer_knn [in] The K-NN of the feature points from the reference image.
	 */
	cv::Mat ComputeFixedKCost(const cv::Mat& query_knn, 
		const cv::Mat& refer_knn) const;

	/**
	 * @brief  Computes the costs using a multi-scale neighborhood representation and determines the optimal inlier set.
	 *
	 * @return void 
	 */
	void ComputeMultiScaleCost();

private:
	const std::vector<cv::Point2d> query_points_;  ///< Points from the query image.
	const std::vector<cv::Point2d> refer_points_;  ///< Points from the reference image.

	const int num_neighbors_;  ///< The number of nearest neighbors for multi-scale neighborhood construction.
	const double lambda_;      ///< Parameter \f$ \lambda\f$ controls the threshold for judging the correctness of a putative correspondence.
	const double tau_;         ///< Parameter \f$ \tau\f$ determines whether a neighboring putative match preserves the consensus of neighborhood topology.

	int num_matches_;   ///< The number of the putative matches.
	
	cv::Mat query_knn_; ///< The K-NN of the feature points from the query image.
	cv::Mat refer_knn_; ///< The K-NN of the feature points from the reference image.

	cv::Mat match_vectors_;  ///< The displacement vectors where the head and tail of each vector correspond to the spatial positions of two corresponding feature points in the two images.
	cv::Mat vector_lengths_; ///< The lengths of the displacement vectors.

	std::vector<bool> labels_; ///< The binary vector that represents the match correctness of the correspondences.
	cv::Mat lpm_cost_;         ///< The costs of the putative matches.
};

#endif



