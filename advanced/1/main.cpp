//
//  main.cpp
//  ImageRetrieval
//
//  Created by Alex Attinger on 13.12.12.
//  Copyright (c) 2012 Alex Attinger. All rights reserved.
//


#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "dirent.h"


using namespace cv;
using namespace std;
bool normalizedHist = false;


Mat getHist(Mat &img){
    
    if(normalizedHist){
        vector<Mat>channels(img.channels());
        split(img,channels);
        for(int i = 0; i<img.channels();i++){
            equalizeHist(channels[i], channels[i]);
        }
        merge(channels,img);
    }
    
    
    Mat hist;
    float range[] = {0,256};
    int channels[] = {0,1,2};
    const float* histRange = {range};
    bool uniform = true, accumulate = false;
    int nBins = 256;
    
    calcHist(&img,1,channels,Mat(),hist,1,&nBins,&histRange,uniform,accumulate);
    
    return hist;
}
//helper function
bool hasEnding (std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

void showNBest(map<double,string> imagescore,int n){
    //show n best matches (lowest scores)
    map<double,string>:: iterator p;
    int i =0;
    for(p=imagescore.begin();p!=imagescore.end();p++){
        cout<<p->first<<endl;
        stringstream title;
        title << "Match no ";
        title << i+1;
        cout<<p->second<<endl;
        imshow(title.str(),imread(p->second));
        i++;
        if(i>=n){
            break;
        }
        
    }
}


int main(int argc, const char * argv[])
{
    //the reference image
    string ref = argv[1];
    Mat refImage, refHist;
    refImage=imread(ref);
    refHist = getHist(refImage);
    imshow("Reference",refImage);
    //the folder containing the test images
    string path = argv[2];
    if(!hasEnding(path, "/")){
        path = path+"/";
    }
    
    DIR *dir;
    struct dirent *ent;
    dir = opendir (argv[2]);
    
    map<double,string> imagescores;
    //values in this map are sorted automatically (lowest first)
    
    
    if (dir != NULL) {
        
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
           //check if image
            string fullname = path+string(ent->d_name);
            
            Mat img = imread(fullname);
            
            if(img.data){
                    
            
                Mat hist = getHist(img);
                int method = CV_COMP_CHISQR;
                double score=compareHist(refHist, hist, method);
                if(method == CV_COMP_CORREL){
                    score = 1.0-score;
                }
                
            
                //CV_COMP_CORREL Correlation need to normalize score values for sorting
                //CV_COMP_CHISQR Chi-Square
                //CV_COMP_INTERSECT Intersection
                //CV_COMP_BHATTACHARYYA Bhattacharyya distance
                imagescores.insert({score,fullname});
                
            }
            
            
        }
        showNBest(imagescores,5);
        closedir (dir);
    } else {
        /* could not open directory */
        perror ("");
        return EXIT_FAILURE;
    }
    
    
    waitKey();
    return 0;
}

