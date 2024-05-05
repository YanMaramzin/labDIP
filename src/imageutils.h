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
Mat gaussDiff(const Mat &gauss1, const Mat &gauss2);
void mosaic(const Mat &inputImg, Mat &outputImg, int maskSize);
void apertureCorrection(const Mat &inputImg, Mat &outputImg, int S);
void medianFilter(const Mat &inputImg, Mat &outputImg);
std::vector<std::vector<double>> gaussMask(const float sigma, int rows, int cols);
int newRangeValue(int oldMin, int oldMax, int newMin, int newMax, int value);
Mat robertX(Mat &image);
Mat robertY(Mat &image);

void erosion(const Mat &input_img, Mat &output_img, size_t size = 3);
void dilation(const Mat &inputImg, Mat &outputImg, size_t size = 3);
void countur(const Mat &inputImg, Mat &outputImg);
Mat opening(const Mat &inputImg, Mat &outputImg);
Mat close(const Mat &inputImg, Mat &outputImg);
Mat multiscaleMorphologicalGradient(const Mat &inputImg);

}
