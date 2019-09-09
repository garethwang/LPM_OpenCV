#include "lpm_matcher.h"
#include "kdtree.h"
#include <fstream>

/**
 * @brief Class for user-defined 2D points to use the KDTree class. 
 */
class Point2D :public cv::Vec2d {
public:
	static const int DIM = 2;
	Point2D(cv::Point2d pt) {
		(*this)[0] = pt.x;
		(*this)[1] = pt.y;
	}
};

/**
 * @brief  Finds K nearest neighbors for the feature points.
 *
 * @return cv::Mat The indices of the K-nearest neighbors found, M*K, CV_32S.
 * @param  features [in] The feature points to index.
 * @param  queries [in] The query points.
 * @param  knn [in] The number of nearest neighbors to search for.
 */
static cv::Mat FindKnnNeighbors(const std::vector<cv::Point2d>& features, 
	const std::vector<cv::Point2d>& queries, const int knn) {

	// Convert the OpenCV point format to the user-defined point format.
	std::vector<Point2D> feature_pts;
	for (int i = 0; i < static_cast<int>(features.size()); ++i) {
		feature_pts.push_back(features[i]);
	}

	// Construct a k-d tree.
	kdt::KDTree<Point2D> kdtree(feature_pts);

	int num_queries = static_cast<int>(queries.size());
	// Perform a K-nearest neighbor search.
	cv::Mat indices(num_queries, knn, CV_32S);
	for (int i = 0; i < num_queries; ++i) {
		Point2D temp_pt(queries[i]);
		std::vector<int> vec_indices = kdtree.knnSearch(temp_pt, knn);
		cv::Mat(vec_indices).reshape(1, 1).copyTo(indices.row(i));
	}

	return indices;
}

LPM_Matcher::LPM_Matcher(const std::vector<cv::Point2d>& query_points, 
	const std::vector<cv::Point2d>& refer_points,
	const int knn, const double lambda, const double tau,
	const std::vector<bool>& labels)
	:query_points_(query_points), refer_points_(refer_points), 
	num_neighbors_(knn), lambda_(lambda), tau_(tau) {

	Initialize(labels);
}

LPM_Matcher::~LPM_Matcher() {}

void LPM_Matcher::Match(cv::Mat& cost, std::vector<bool>& labels) {

	ComputeMultiScaleCost();
	lpm_cost_.copyTo(cost);
	labels = labels_;
}

void LPM_Matcher::Initialize(const std::vector<bool>& labels) {

	CV_Assert(query_points_.size() == refer_points_.size());
	num_matches_ = static_cast<int>(query_points_.size());
	cv::Mat queries = cv::Mat(query_points_).reshape(1);
	cv::Mat referes = cv::Mat(refer_points_).reshape(1);

	// Convert the putative matches into displacement vectors.
	match_vectors_ = referes - queries;
	vector_lengths_ = match_vectors_.col(0).mul(match_vectors_.col(0)) +
		match_vectors_.col(1).mul(match_vectors_.col(1));
	cv::sqrt(vector_lengths_, vector_lengths_);

	int knn = num_neighbors_ + 1;

	// Find the k-nearest neighbor of the feature points.
	cv::Mat query_knn(num_matches_, knn, CV_32S);
	cv::Mat	refer_knn(num_matches_, knn, CV_32S);
	if (labels.empty()) {
		query_knn = FindKnnNeighbors(query_points_, query_points_, knn);
		refer_knn = FindKnnNeighbors(refer_points_, refer_points_, knn);
	}
	else { // Construct the neighborhoods based on the inlier set.
		std::vector<cv::Point2d> query_inliers, refer_inliers;
		std::vector<int> inlier_indices;
		for (int i = 0; i < num_matches_; ++i) {
			if (labels[i]) {
				inlier_indices.push_back(i);
				query_inliers.push_back(query_points_[i]);
				refer_inliers.push_back(refer_points_[i]);
			}
		}

		cv::Mat inliers_knn0, inliers_knn1;
		inliers_knn0 = FindKnnNeighbors(query_inliers, query_points_, knn);
		inliers_knn1 = FindKnnNeighbors(refer_inliers, refer_points_, knn);

		for (int i = 0; i < inliers_knn0.rows; ++i) {
			int* pknn0 = (int*)inliers_knn0.ptr(i);
			int* pknn1 = (int*)inliers_knn1.ptr(i);

			int* pqk = (int*)query_knn.ptr(i);
			int* prk = (int*)refer_knn.ptr(i);
			for (int j = 0; j < inliers_knn0.cols; ++j) {
				pqk[j] = inlier_indices[pknn0[j]];
				prk[j] = inlier_indices[pknn1[j]];
			}
		}
	}

	// Delete the first column, which is the nearest neighbor, i.e. the feature point itself.
	query_knn.colRange(1, query_knn.cols).copyTo(query_knn_);
	refer_knn.colRange(1, refer_knn.cols).copyTo(refer_knn_);
}

std::vector<std::vector<int> > LPM_Matcher::FindNeighborsIntersection(
	const cv::Mat& query_knn, const cv::Mat& refer_knn) const {
	
	std::vector<std::vector<int> > consensus(num_matches_);
	for (int i = 0; i < num_matches_; ++i) {	
		std::vector<int> vec0(query_knn.row(i).reshape(1));
		std::vector<int> vec1(refer_knn.row(i).reshape(1));

		std::sort(vec0.begin(), vec0.end());
		std::sort(vec1.begin(), vec1.end());

		std::vector<int> vec_intersection;
		std::set_intersection(vec0.begin(), vec0.end(), vec1.begin(), vec1.end(), 
			std::back_inserter(vec_intersection));

		consensus[i] = vec_intersection;
	}

	return consensus;
}

cv::Mat LPM_Matcher::ComputeFixedKCost(const cv::Mat& query_knn, 
	const cv::Mat& refer_knn) const {

	cv::Mat cost(num_matches_, 1, CV_64F);
	double* pcost = (double*)cost.data;

	int knn = query_knn.cols;
	double inv_knn = 1.0 / knn;

	std::vector<std::vector<int> > knn_intersection;
	knn_intersection = FindNeighborsIntersection(query_knn, refer_knn);

	// The indices of common elements in the two neighborhoods for the ith match.
	std::vector<int> indices_knni;
	double* pdis = (double*)vector_lengths_.data;
	for (int i = 0; i < num_matches_; ++i) {
		indices_knni = knn_intersection[i];
		// The number of common elements in the K-NN.
		size_t num_inter = indices_knni.size();
		
		double c1 = static_cast<double>(knn - num_inter); // K-ni

		double c2 = 0;
		double* pveci = (double*)match_vectors_.ptr(i);
		for (size_t j = 0; j < num_inter; ++j) {
			double* pvecj = (double*)match_vectors_.ptr(indices_knni[j]);
			//               (vi,vj)
			// cos(theta) = ---------    Eq.(9)
			//               |vi||vj|
			double cos_theta = (pveci[0] * pvecj[0] + pveci[1] * pvecj[1]) /
				(pdis[i] * pdis[indices_knni[j]]);

			//          min{|vi|,|vj|}
			// ratio = ----------------   Eq.(9)
			//          max{|vi|,|vj|}
			double ratio = std::min(pdis[i], pdis[indices_knni[j]]) /
				std::max(pdis[i], pdis[indices_knni[j]]);

			if (ratio*cos_theta < tau_) {
				c2++;
			}
		}
		pcost[i] = (c1 + c2) * inv_knn;
	}

	return cost;
}

void LPM_Matcher::ComputeMultiScaleCost() {

	// Computes the costs according to Eq.(14).
	lpm_cost_ = cv::Mat::zeros(num_matches_, 1, CV_64F);
	int num_scales = 3;

	for (int i = 0; i < num_scales; ++i) {
		cv::Mat temp_knn0(query_knn_.colRange(i, num_neighbors_ - i));
		cv::Mat temp_knn1(refer_knn_.colRange(i, num_neighbors_ - i));

		cv::Mat temp_cost = ComputeFixedKCost(temp_knn0, temp_knn1);
		lpm_cost_ += temp_cost;
	}

	lpm_cost_ /= num_scales;

	labels_.resize(num_matches_);
	double* pcost = (double*)lpm_cost_.data;
	for (int i = 0; i < num_matches_; ++i) {		
		if (pcost[i] <= lambda_)
			labels_[i] = 1; // pi=1 if ci<=\f$ \lambda\f$
		else
			labels_[i] = 0; // pi=0 if ci>\f$ \lambda\f$
	}
}
