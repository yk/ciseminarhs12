#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

void showandsave(string name, Mat img) {
	imwrite("data/res/" + name + ".png", img);
	imshow(name, img);
	waitKey();
}

void siftandsurf() {
	Mat edges = imread("data/library.jpg", CV_LOAD_IMAGE_GRAYSCALE);

//	SiftFeatureDetector detector(0.15,0.15);
	SurfFeatureDetector detector(4000);
	vector<KeyPoint> keypoints;
	detector.detect(edges, keypoints);
//	SiftDescriptorExtractor extractor;
	SurfDescriptorExtractor extractor;
	Mat descriptors;
	extractor.compute(edges, keypoints, descriptors);
	drawKeypoints(edges, keypoints, edges);
	showandsave("surfedges", edges);
}

vector<KeyPoint> getHarrisPoints(Mat& img,int thresh=95) {
	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(img.size(), CV_32FC1);
//	int thresh = 95;

	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	/// Detecting corners
	cornerHarris(img, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	vector<KeyPoint> v;
	/// Drawing a circle around corners
	for (int j = 0; j < dst_norm.rows; j++) {
		for (int i = 0; i < dst_norm.cols; i++) {
			if ((int) dst_norm.at<float>(j, i) > thresh) {
				v.push_back(KeyPoint(i,j,1));
			}
		}
	}
	cout << "got " << v.size() << " harris points" << endl;
	return v;
}

void harris() {
	Mat src_gray = imread("data/bookT.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src_gray.size(), CV_32FC1);
//	int thresh = 95; //libr
//	int thresh = 200; //r
//	int thresh = 190; //p
	int thresh = 130; //t

	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	/// Detecting corners
	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	/// Drawing a circle around corners
	for (int j = 0; j < dst_norm.rows; j++) {
		for (int i = 0; i < dst_norm.cols; i++) {
			if ((int) dst_norm.at<float>(j, i) > thresh) {
				circle(src_gray, Point(i, j), 5, Scalar(0), 2, 8, 0);
//				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
			}
		}
	}
	/// Showing the result
	showandsave("harris_bookt", src_gray);
}

void match() {
	Mat img = imread("data/library.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat book = imread("data/book.jpg", CV_LOAD_IMAGE_GRAYSCALE);

//	int minHessian = 400;

	SurfFeatureDetector detector(2500);
//	SiftFeatureDetector detector(0.14,0.14);

	std::vector<KeyPoint> keypoints_1, keypoints_2;

	detector.detect(img, keypoints_1);
	detector.detect(book, keypoints_2);

//	keypoints_1 = getHarrisPoints(img);
//	keypoints_2 = getHarrisPoints(book,130);

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;
//	SiftDescriptorExtractor extractor;

	Mat descriptors_1, descriptors_2;

	extractor.compute(img, keypoints_1, descriptors_1);
	extractor.compute(book, keypoints_2, descriptors_2);

//	cout << "d1 " << descriptors_1 << " d2 " << descriptors_2 << endl;

	//-- Step 3: Matching descriptor vectors with a brute force matcher
	BruteForceMatcher<L2<float> > matcher;
	std::vector<DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	float thresh = 0.2;

	auto newend =
			remove_if(matches.begin(), matches.end(),
					[&](DMatch& match)->bool {/*cout<<match.distance<<endl;*/return (abs(match.distance) > thresh);});
	auto good_matches = vector<DMatch>(matches.begin(), newend);
	//-- Draw matches
	Mat img_matches;
	drawMatches(img, keypoints_1, book, keypoints_2, good_matches, img_matches);

	//-- Show detected matches
	showandsave("Matches_surf_02", img_matches);
}

float distPointLine(float a, float b, float x, float y) {
	return abs(a * x - y + b) / sqrt(a * a + 1);
}

float ran(float min, float max) {
	float r3 = min + (float) rand() / ((float) RAND_MAX / (max - min));
	return r3;
}

void ransac() {
#include "data/data.hpp"
	float a, b;
	float mina = -1.0, maxa = 1.0, minb = 5.0, maxb = 10.0;
	int numCons;
	float boundary = 2.7;
	float consRatio = 0.7;
	do {
		a = ran(mina, maxa);
		b = ran(minb, maxb);
		numCons =
				count_if(data.begin(), data.end(),
						[&](Point2f& p)->bool {return (distPointLine(a,b,p.x,p.y)<boundary);});
		cout << "nc " << numCons << " ds " << data.size() << " r "
				<< numCons * 1.0 / data.size() << " cr " << consRatio << " b "
				<< (numCons * 1.0 / data.size() < consRatio) << endl;
	} while ((numCons * 1.0 / data.size()) < consRatio);
	cout << "a = " << a << ", b = " << b << endl;
}

int main() {
//	siftandsurf();
//	harris();
//	match();
	ransac();
}

