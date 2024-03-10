//
// Created by mist on 19.02.24.
//
#include "imageutils.h"
using cv::Vec3b;

Mat ImageUtils::monochromeImage(const Mat &image)
{
    Mat resultImage = Mat::zeros(image.rows, image.cols, CV_8UC1);
    imageProcessing(image, [image, &resultImage](int i, int j)
    {
        resultImage.at<uchar>(i, j) = 0.36 * image.at<Vec3b>(i, j)[2] + 0.53 * image.at<Vec3b>(i, j)[1] +
                                      0.11 * image.at<Vec3b>(i, j)[0];
    });

    return resultImage;
}

Mat ImageUtils::hist(const cv::Mat &image)
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

Mat ImageUtils::quantizedImage(Mat &image, const int quantLevel)
{
    Mat resultImage = Mat::zeros(image.rows, image.cols, CV_8UC1);
    auto intervalQuant = 255.0 / (quantLevel - 1);
    imageProcessing(image, [intervalQuant, &image, &resultImage](int i, int j)
    {
        resultImage.at<uchar>(i, j) = round(image.at<uchar>(i, j) / intervalQuant) * intervalQuant;
    });
    return resultImage;
}

void ImageUtils::imageProcessing(const Mat &image, std::function<void(int i, int j)> func)
{
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            func(i, j);
        }
    }
}

Mat ImageUtils::basisMatrix(int rows, int cols)
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

Mat ImageUtils::discretCosineTransform(const Mat &image)
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
