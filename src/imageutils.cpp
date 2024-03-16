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

    for (int i = 0; i < inputImg.cols; ++i)
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

void mosaic(const Mat &inputImg, Mat &outputImg)
{

}

std::vector<std::vector<double>> gaussMask(const float sigma, int rows, int cols)
{
    std::vector<std::vector<double>> mask(rows, std::vector<double>(cols));
    int xMin = (rows + 1) / 2 - rows;
    int xMax = rows - (rows + 1) / 2;
    int yMin = (cols + 1) / 2 - cols;
    int yMax = cols - (cols + 1) / 2;

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

}