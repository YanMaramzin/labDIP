//
// Created by mist on 19.02.24.
//
#include "imageutils.h"

using cv::Vec3b;

void tmp(const Mat &image, std::function<void(int i, int j)> func);

void tmp(const Mat &image, std::function<void(int i, int j)> func)
{
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            func(i, j);
        }
    }
}

Mat ImageUtils::monochromeImage(const Mat &image)
{
    Mat resultImage = Mat::zeros(image.rows, image.cols, CV_8UC1);
    // for (int i = 0; i < image.rows; ++i) {
    //     for (int j = 0; j < image.cols; ++j) {
    // resultImage.at<uchar>(i, j) = 0.36 * image.at<Vec3b>(i, j)[2] + 0.53 * image.at<Vec3b>(i, j)[1] +
    //         0.11 * image.at<Vec3b>(i, j)[0];
    //     }
    // }
    tmp(image, [image, &resultImage](int i, int j)
    {
        resultImage.at<uchar>(i, j) = 0.36 * image.at<Vec3b>(i, j)[2] + 0.53 * image.at<Vec3b>(i, j)[1] +
                0.11 * image.at<Vec3b>(i, j)[0];
    });

    return resultImage;
}

Mat ImageUtils::hist(const cv::Mat &image)
{
    Mat hist = Mat::zeros(1, 256,CV_64FC1);
    // for (int i = 0; i < image.rows; ++i) {
    //     for (int j = 0; j < image.cols; ++j) {
    // int r = image.at<uchar>(j, i);
    // hist.at<double>(0, r) = hist.at<double>(0, r) + 1.0;
    //     }
    // }
    tmp(image, [image, &hist](int i, int j)
    {
        int r = image.at<uchar>(j, i);
        hist.at<double>(0, r) = hist.at<double>(0, r) + 1.0;
    });

    double min = 0;
    double max = 0;
    cv::minMaxLoc(hist, &min, &max);
    hist /= max;

    Mat histImg = Mat::zeros(100, 256, CV_8U);

    // for (int i = 0; i < 256; ++i) {
    //     for (int j = 0; j < 100; ++j) {
    // if (hist.at<double>(0, i) * 100 > j) {
    //     histImg.at<uchar>(99 - j, i) = 255;
    // }
    //     }
    // }
    tmp(histImg, [&histImg, &hist](int i, int j)
    {
        if (hist.at<double>(0, j) * 100 > i)
            histImg.at<uchar>(99 - i, j) = 255;
    });
    cv::bitwise_not(histImg, histImg);
    return histImg;
}

Mat ImageUtils::quantizedImage(const Mat &image, const int intervalQuant)
{
    (void)intervalQuant;
    return Mat();
}

Mat ImageUtils::showImages(std::string &title, int imageCount, ...)
{
    return Mat();
}
