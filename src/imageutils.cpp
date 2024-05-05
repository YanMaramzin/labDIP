//
// Created by mist on 19.02.24.
//
#include <numeric>
#include <iostream>
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

    for(int n = 0; n < basisMatrix.rows; ++n)
        for(int k = 0; k < basisMatrix.cols; ++k) {
            if (n == 0)
                basisMatrix.at<double>(n, k) = std::sqrt(1.0 / basisMatrix.cols);
            else
                basisMatrix.at<double>(n, k) = std::sqrt(2.0 / basisMatrix.cols) * std::cos(M_PI * n * (k + 1.0 / 2) / basisMatrix.cols);
        }

    return basisMatrix.t();
}

Mat discretCosineTransform(const Mat &image)
{
    Mat basMat = basisMatrix(image.cols, image.cols);
    Mat basMatTransp = basMat.t();
    Mat imageBlock64F;
    image.convertTo(imageBlock64F, CV_64F);
    Mat matMultiply = basMatTransp * imageBlock64F;
    Mat DCT = matMultiply * basMat;
    Mat DCT8U;
    DCT.convertTo(DCT8U, CV_8U);

    return DCT8U;
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

void erosion(const Mat &input_img, Mat &output_img, size_t size)
{
    const int xMin = (size + 1) / 2 - size;
    const int xMax = size - (size + 1) / 2;
    const int yMin = (size + 1) / 2 - size;
    const int yMax = size - (size + 1) / 2;

    output_img = Mat::zeros(input_img.size(), CV_8U);
    for (int i = 1; i < input_img.cols - 1; ++i)
        for (int j = 1; j < input_img.rows - 1; ++j) {
            float min = 255;
            for (int ii = xMin; ii <= xMax; ++ii)
                for (int jj = yMin; jj <= yMax; ++jj) {
                    const uchar Y = input_img.at<uchar>(j + jj, i + ii);
                    if (Y > min)
                        continue;
                    min = Y;
                }
            output_img.at<uchar>(j, i) = min;
        }
}

void dilation(const Mat &inputImg, Mat &outputImg, size_t size)
{
    const int xMin = (size + 1) / 2 - size;
    const int xMax = size - (size + 1) / 2;
    const int yMin = (size + 1) / 2 - size;
    const int yMax = size - (size + 1) / 2;

    outputImg = Mat::zeros(inputImg.size(), CV_8U);
    for (int i = 1; i < inputImg.cols - 1; ++i)
        for (int j = 1; j < inputImg.rows - 1; ++j) {
            float max = 0;
            for (int ii = xMin; ii <= xMax; ++ii)
                for (int jj = yMin; jj <= yMax; ++jj) {
                    const uchar Y = inputImg.at<uchar>(j + jj, i + ii);
                    if (Y < max)
                        continue;
                    max = Y;
                }
            outputImg.at<uchar>(j, i) = max;
        }
}

void countur(const Mat &inputImg, Mat &outputImg)
{
    Mat tmp;
    erosion(inputImg, tmp);
    outputImg = inputImg - tmp;
}

Mat opening(const Mat &inputImg, Mat &outputImg)
{
    erosion(inputImg, outputImg);
    Mat tmp;
    dilation(outputImg, tmp);
    return tmp;
}

Mat close(const Mat &inputImg, Mat &outputImg)
{
    dilation(inputImg, outputImg);
    Mat tmp;
    erosion(outputImg, tmp);
    return tmp;
}


void apertureCorrection(const Mat &inputImg, Mat &outputImg, int S)
{
    outputImg = Mat::zeros(inputImg.size(), CV_8U);
    int X = ceil((100.0 / (S - 1)) + 8);
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
    constexpr int num = 3 * 3;
    std::vector<uchar> pixel(num);

    constexpr int delta[3 * 3][2] = {
        {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}
    };
    for (int i = 1; i < inputImg.rows - 1; ++i) {
        for (int j = 1; j < inputImg.cols - 1; ++j) {
            for (int k = 0; k < num; ++k)
                pixel[k] = inputImg.at<uchar>(i + delta[k][0], j + delta[k][1]);
            std::sort(pixel.begin(), pixel.end());
            outputImg.at<uchar>(i, j) = pixel[num / 2];
        }
    }
}

Mat robertX(Mat &image)
{
    int kernelX[2][2] = {{1, 0}, {0, -1}};
    Mat result = Mat::zeros(image.size(), image.type());

    for (int y = 0; y < image.rows - 1; ++y) {
        for (int x = 0; x < image.cols - 1; ++x) {
            int sum = 0;

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j)
                    sum += kernelX[i][j] * image.at<uchar>(y + i, x + j);
            }

            result.at<uchar>(y, x) = abs(sum);
        }
    }

    return result;
}

Mat robertY(Mat &image)
{
    int kernelX[2][2] = {{0, 1}, {-1, 0}};
    Mat result = Mat::zeros(image.size(), image.type());

    for (int y = 0; y < image.rows - 1; ++y) {
        for (int x = 0; x < image.cols - 1; ++x) {
            int sum = 0;

            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j)
                    sum += kernelX[i][j] * image.at<uchar>(y + i, x + j);
            }

            result.at<uchar>(y, x) = abs(sum);
        }
    }

    return result;
}

Mat multiscaleMorphologicalGradient(const Mat &inputImg)
{
    Mat outputImg = Mat::zeros(inputImg.size(), CV_8U);
    for (int i = 1; i <= 3; ++i) {
        Mat tmpDilation = Mat::zeros(inputImg.size(), CV_8U);
        Mat tmpErosion = Mat::zeros(inputImg.size(), CV_8U);
        Mat tmp = Mat::zeros(inputImg.size(), CV_8U);
        dilation(inputImg, tmpDilation, 2*i + 1);
        erosion(inputImg, tmpErosion, 2*i + 1);
        Mat raz = tmpDilation - tmpErosion;
        erosion(raz, tmp, 2 * (i - 1) + 1);
        outputImg += tmp;
    }

    return outputImg / 3;
}

}
