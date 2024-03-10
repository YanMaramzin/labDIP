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
    Mat quantizedImage(Mat &image, const int quantLevel);
    void imageProcessing(const Mat &image, std::function<void(int i, int j)> func);
    Mat basisMatrix(int rows, int cols);
    Mat discretCosineTransform(const Mat &image);

}