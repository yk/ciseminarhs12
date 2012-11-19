#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

int edgeandsurf() {
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	Mat edges;
	namedWindow("edges", 1);
	for (;;) {
		Mat frame;
		cap >> frame; // get a new frame from camera
		cvtColor(frame, edges, CV_BGR2GRAY);
		GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
		Canny(edges, edges, 0, 30, 3);
//		SurfFeatureDetector detector(500);
//		vector<KeyPoint> keypoints;
//		detector.detect(edges,keypoints);
//		SurfDescriptorExtractor extractor;
//		Mat descriptors;
//		extractor.compute(edges,keypoints,descriptors);
//		drawKeypoints(edges,keypoints,edges);
		imshow("edges", edges);
		if (waitKey(30) >= 0)
			break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}

int matchimages() {
	Mat img_1 = imread("data/lenaeye.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_2 = imread("data/lena.png", CV_LOAD_IMAGE_GRAYSCALE);

	if (!img_1.data || !img_2.data) {
		return -1;
	}

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;

	SurfFeatureDetector detector(minHessian);

	std::vector<KeyPoint> keypoints_1, keypoints_2;

	detector.detect(img_1, keypoints_1);
	detector.detect(img_2, keypoints_2);

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;

	Mat descriptors_1, descriptors_2;

	extractor.compute(img_1, keypoints_1, descriptors_1);
	extractor.compute(img_2, keypoints_2, descriptors_2);

	//-- Step 3: Matching descriptor vectors with a brute force matcher
	BruteForceMatcher<L2<float> > matcher;
	std::vector<DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	//-- Draw matches
	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);

	//-- Show detected matches
	imshow("Matches", img_matches);

	waitKey(0);

	return 0;
}

int matchvideotoimage() {
	Mat img_1 = imread("data/vetter.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	int minHessian = 400;

	SurfFeatureDetector detector(minHessian);

	std::vector<KeyPoint> keypoints_1;

	detector.detect(img_1, keypoints_1);

	SurfDescriptorExtractor extractor;

	Mat descriptors_1;

	extractor.compute(img_1, keypoints_1, descriptors_1);

	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	Mat edges;
	namedWindow("edges", 1);
	for (;;) {
		Mat frame;
		cap >> frame; // get a new frame from camera
		cvtColor(frame, edges, CV_BGR2GRAY);
//			GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
//			Canny(edges, edges, 0, 30, 3);
		SurfFeatureDetector detector(500);
		vector<KeyPoint> keypoints;
		detector.detect(edges, keypoints);
		SurfDescriptorExtractor extractor2;
		Mat descriptors;
		extractor.compute(edges, keypoints, descriptors);
//			drawKeypoints(edges,keypoints,edges);
//		BruteForceMatcher<L2<float> > matcher;
		FlannBasedMatcher matcher;
		std::vector<DMatch> matches;
		matcher.match(descriptors_1, descriptors, matches);


		  double max_dist = 0; double min_dist = 100;

		  //-- Quick calculation of max and min distances between keypoints
		  for( int i = 0; i < descriptors_1.rows; i++ )
		  { double dist = matches[i].distance;
		    if( dist < min_dist ) min_dist = dist;
		    if( dist > max_dist ) max_dist = dist;
		  }

		  //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
		  //-- PS.- radiusMatch can also be used here.
		  std::vector< DMatch > good_matches;

		  for( int i = 0; i < descriptors_1.rows; i++ )
		  { if( matches[i].distance < 2*min_dist )
		    { good_matches.push_back( matches[i]); }
		  }




		//-- Draw matches
		Mat img_matches;
		drawMatches(img_1, keypoints_1, edges, keypoints, good_matches, img_matches);
		imshow("matches", img_matches);
		if (waitKey(30) >= 0)
			break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}

int main() {

//	return edgeandsurf();
//	return matchimages();
	return matchvideotoimage();
}
