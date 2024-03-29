//
// Created by mist on 19.02.24.
//
#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <memory>

using cv::Mat;

namespace ImageUtils
{

class ImageHandler
{
public:
    ImageHandler();
    ~ImageHandler();

private:
    struct Pimpl;
    std::unique_ptr<Pimpl> m_d;
};
Mat monochromeImage(const Mat &image);
Mat hist(const Mat &image);
Mat quantizedImage(Mat &image, const int quantLevel);
void imageProcessing(const Mat &image, std::function<void(int i, int j)> func);
Mat basisMatrix(int rows, int cols);
Mat discretCosineTransform(const Mat &image);
double sko(const Mat &image, const Mat quantImage);
void gauss(const Mat &inputImg, Mat &outputImg, int maskSize, float sko);
void mosaic(const Mat &inputImg, Mat &outputImg);
std::vector<std::vector<double>> gaussMask(const float sigma, int rows, int cols);
int newRangeValue(int oldMin, int oldMax, int newMin, int newMax, int value);
}