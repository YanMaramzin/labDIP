//
// Created by mist on 19.02.24.
//
#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using cv::Mat;

namespace ImageUtils
{
    Mat monochromeImage(const Mat &image);
    Mat hist(const Mat &image);
    Mat quantizedImage(const Mat &image);
    Mat showImages(std::string &title, int imageCount, ...);
    
}