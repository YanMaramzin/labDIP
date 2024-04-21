//
// Created by mist on 19.02.24.
//
#include <numeric>
#include "imageutils.h"
using cv::Vec3b;

namespace ImageUtils {
struct ImageHandler::Pimpl
{

};

ImageHandler::ImageHandler() :
    m_d(std::make_unique<Pimpl>())
{}

ImageHandler::~ImageHandler()
{}


Mat monochromeImage(const Mat &image)
{
    Mat resultImage = Mat::zeros(image.rows, image.cols, CV_8UC1);
    imageProcessing(image, [image, &resultImage](int i, int j)
    {
        resultImage.at<uchar>(i, j) = 0.36 * image.at<Vec3b>(i, j)[2] + 0.53 * image.at<Vec3b>(i, j)[1] +
                                      0.11 * image.at<Vec3b>(i, j)[0];
    });

    return resultImage;
}

Mat hist(const cv::Mat &image)
{
    Mat hist = Mat::zeros(1, 256,CV_64FC1);
    imageProcessing(image, [image, &hist](int i, int j)
    {
        int r = image.at<uchar>(j, i);
        hist.at<double>(0, r) = hist.at<double>(0, r) + 1.0;
    });

    double min = 0;
    double max = 0;
    cv::minMaxLoc(hist, &min, &max);
    hist /= max;

    Mat histImg = Mat::zeros(100, 256, CV_8U);
    imageProcessing(histImg, [&histImg, &hist](int i, int j)
    {
        if (hist.at<double>(0, j) * 100 > i)
            histImg.at<uchar>(99 - i, j) = 255;
    });
    cv::bitwise_not(histImg, histImg);
    return histImg;
}

Mat quantizedImage(Mat &image, const int quantLevel)
{
    Mat resultImage = Mat::zeros(image.rows, image.cols, CV_8UC1);
    auto intervalQuant = 255.0 / (quantLevel - 1);
    imageProcessing(image, [intervalQuant, &image, &resultImage](int i, int j)
    {
        resultImage.at<uchar>(i, j) = round(image.at<uchar>(i, j) / intervalQuant) * intervalQuant;
    });
    return resultImage;
}

void imageProcessing(const Mat &image, std::function<void(int i, int j)> func)
{
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            func(i, j);
        }
    }
}

Mat basisMatrix(int rows, int cols)
{
    Mat basisMatrix = Mat::zeros(rows, cols, CV_64F);

    imageProcessing(basisMatrix, [&basisMatrix](int n, int k)
    {
        if (n == 0)
            basisMatrix.at<double>(n, k) = std::sqrt(1.0f / basisMatrix.rows);
        else
            basisMatrix.at<double>(n, k) = sqrt(2.0 / basisMatrix.rows) * cos((M_PI * n) * (k + 1.0f / 2) / basisMatrix.rows);
    });

    return basisMatrix;
}

Mat discretCosineTransform(const Mat &image)
{
    Mat basMat = basisMatrix(image.rows, image.cols);
    Mat basMatTransp = basMat.t();
    Mat imageBlock64F;
    image.convertTo(imageBlock64F, CV_64F);
    Mat matMultiply = basMatTransp * imageBlock64F;
    Mat DCT = matMultiply * basMat;
    DCT.convertTo(DCT, CV_8U);

    return DCT;
}

double sko(const Mat &image, const Mat quantImage)
{
    double sum {0};
    imageProcessing(image, [&image, &quantImage, &sum](int i, int j)
    {
        sum += std::pow(image.at<uchar>(i, j) - quantImage.at<uchar>(i, j),2);
    });
    double sko = sqrt(1.0 / (image.rows * image.cols) * sum);
    return  sko;
}

void gauss(const Mat &inputImg, Mat &outputImg, int maskSize, float sko)
{
    outputImg = Mat::zeros(inputImg.size(), CV_8U);
    auto F = gaussMask(sko, maskSize, maskSize);
    int maskMin = (maskSize + 1) / 2 - maskSize;
    int maskMax = maskSize - (maskSize + 1) / 2;

    for (int i = 0; i < inputImg.cols; ++i) {
        for (int j = 0; j < inputImg.rows; ++j) {
            float Rez = 0.0f;
            for (int ii = maskMin; ii <= maskMax; ++ii)
                for (int jj = maskMin; jj <= maskMax; ++jj) {
                    uchar blurred = inputImg.at<uchar>(j + jj, i + ii);
                    Rez += F[ii + maskMax][jj + maskMax] * blurred;
                }
            outputImg.at<uchar>(j, i) = Rez;
        }
    }
}

Mat gaussDiff(const Mat &gauss1, const Mat &gauss2)
{
    Mat output;
    cv::absdiff(gauss1, gauss2, output);
    return output;
}

void mosaic(const Mat &inputImg, Mat &outputImg, int maskSize)
{
    outputImg = Mat::zeros(inputImg.size(), CV_8U);
    const auto maskMin = (maskSize + 1) / 2 - maskSize;
    const auto maskMax = maskSize - (maskSize + 1) / 2;
    auto pp = 1.0f / 9;
    std::vector<std::vector<float>> mask {{pp, pp, pp},
                                      {pp, pp, pp},
                                      {pp, pp, pp}};

     for (int i = 0; i < inputImg.cols; i += maskSize) {
         for (int j = 0; j < inputImg.rows; j += maskSize) {
             auto Rez {0.0f};
             for (int ii = maskMin; ii <= maskMax; ++ii)
                 for (int jj = maskMin; jj <= maskMax; ++jj) {
                     uchar blurred = inputImg.at<uchar>(j + jj, i + ii);
                     Rez += mask[ii + maskMax][jj + maskMax] * blurred;
                 }
             for (auto k = i; k <= i + maskSize; ++k) {
                 for (auto l = j; l <= j + maskSize; ++l)
                     outputImg.at<uchar>(l, k) = Rez;
             }
         }
     }
}

std::vector<std::vector<double>> gaussMask(const float sigma, int rows, int cols)
{
    std::vector<std::vector<double>> mask(rows, std::vector<double>(cols));
    const int xMin = (rows + 1) / 2 - rows;
    const int xMax = rows - (rows + 1) / 2;
    const int yMin = (cols + 1) / 2 - cols;
    const int yMax = cols - (cols + 1) / 2;

    for (size_t i = 0; i < rows; ++i) {
        int x = newRangeValue(0, rows - 1, xMin, xMax, i);
        for (size_t j = 0; j < cols; ++j) {
            int y = newRangeValue(0, cols - 1, yMin, yMax, j);
            mask[i][j] = exp(-(pow(x, 2) + pow(y,2)) / (2 * pow(sigma, 2))) / (2 * M_PI * pow(sigma, 2));
        }
    }
    return mask;
}

int newRangeValue(int oldMin, int oldMax, int newMin, int newMax, int value)
{
    return (value - oldMin) * (newMax - newMin) / (oldMax - oldMin) + newMin;
}

void apertureCorrection(const Mat &inputImg, Mat &outputImg, int S)
{
    outputImg = Mat::zeros(inputImg.size(), CV_8U);
    int X = ceil((100.0 / S - 1) + 8);
    auto sum = 8 * (-1) + X;
    auto pp = -1.0f / sum;
    float newX = X / sum;
    std::vector<std::vector<float>> F {{pp , pp, pp},
                                       {pp, newX, pp},
                                       {pp, pp, pp} };

    for (int i = 0; i < inputImg.cols; ++i) {
        for (int j = 0; j < inputImg.rows; ++j) {
            float Rez = 0.0f;
            for (int ii = -1; ii <= 1; ++ii)
                for (int jj = -1; jj <= 1; ++jj) {
                    uchar blurred = inputImg.at<uchar>(j + jj, i + ii);
                    Rez += F[ii + 1][jj + 1] * blurred;
                }
            outputImg.at<uchar>(j, i) = Rez;
        }
    }
}

void medianFilter(const Mat &inputImg, Mat &outputImg)
{
    outputImg = inputImg.clone();
    //0. Preparation: Get the width, height and pixel information of the picture,
    constexpr int num = 3 * 3;
    std::vector<uchar> pixel(num);

    //Relative to the center point, the position where the point in the 3*3 area needs to be offset
    constexpr int delta[3 * 3][2] = {
        {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}
    };
    //1. Median filtering, without considering edges
    for (int i = 1; i < inputImg.rows - 1; ++i) {
        for (int j = 1; j < inputImg.cols - 1; ++j) {
            //1.1 Extract the field value // Use an array to deal with 8 neighborhood values ​​like this is not suitable for larger windows
            for (int k = 0; k < num; ++k) {
                pixel[k] = inputImg.at<uchar>(i + delta[k][0], j + delta[k][1]);
            }
            //1.2 Sorting // Use the built-in library and sorting
            std::sort(pixel.begin(), pixel.end());
            //1.3 Get the value of the center point
            outputImg.at<uchar>(i, j) = pixel[num / 2];
        }
    }
}
}
