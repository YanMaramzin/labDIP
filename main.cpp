#include <iostream>
#include "imageutils.h"

int blockSize = 200;

static void onMouse(int event, int x, int y, int, void *param)
{
    auto img = static_cast<Mat*>(param);
    if (event != cv::EVENT_LBUTTONDOWN)
        return;

    int blockX = x - x % blockSize;
    int blockY = y - y % blockSize;

    cv::Rect block2 = cv::Rect(blockX, blockY, blockSize, blockSize);
    Mat imageBlock = cv::Mat(*img, block2);
    auto DCT = ImageUtils::discretCosineTransform(imageBlock);
    cv::imshow("DCT", DCT);

    auto hist = ImageUtils::hist(DCT);
    cv::imshow("Гистограмма", hist);

    cv::imshow("Выбранный блок", imageBlock);
    cv::Rect block = cv::Rect(blockX, blockY, blockSize, blockSize);
    rectangle(*img, block, cv::Scalar(1, 255, 255));
    cv::imshow("Сетка", *img);

    std::cout << "X: " << blockX << std::endl;
    std::cout << "Y: " << blockY << std::endl;
}

void lab1()
{
    auto width = 400;
    auto heigth = 400;
    auto image = cv::imread("/home/mist/Downloads/zoro.jpeg");
    auto imageMono = ImageUtils::monochromeImage(image);
    auto histImg = ImageUtils::hist(imageMono);
    resize(imageMono, imageMono, cv::Size(width, heigth));
    resize(histImg, histImg, cv::Size(width, heigth));
    Mat combinedImage;
    hconcat(imageMono, histImg, combinedImage);
    auto quant = ImageUtils::quantizedImage(imageMono, 2);
    auto quantHist = ImageUtils::hist(imageMono);
    resize(quant, quant, cv::Size(width, heigth));
    resize(quantHist, quantHist, cv::Size(width, heigth));
    Mat combinedImage2;
    hconcat(quant, quantHist, combinedImage2);
    Mat combinedImage3;
    cv::vconcat(combinedImage, combinedImage2, combinedImage3);
    imshow("После", combinedImage3);
    cv::waitKey();
}

void lab2()
{
    auto image = cv::imread("/home/mist/Downloads/zoro.jpeg");
    auto imageMono = ImageUtils::monochromeImage(image);
    Mat imageWithBlocks = imageMono.clone();

    for (int row = 0; row < imageMono.rows; row += blockSize) {
        for (int col = 0; col < imageMono.cols; col += blockSize) {
            cv::Rect block = cv::Rect(col, row, blockSize, blockSize);
            rectangle(imageWithBlocks, block, cv::Scalar(0, 0, 0));
        }
    }

    cv::imshow("Сетка", imageWithBlocks);
    cv::setMouseCallback("Сетка", onMouse, &imageWithBlocks);

    cv::waitKey();
}

void lab3()
{
    std::cout << "Пока в процессе";
}

int main(int argc, char *argv[])
{
    for (int i = 0; i < argc; ++i) {
        auto arg = std::string(argv[i]);
        if (arg == "-l1")
            lab1();
        else if (arg == "-l2")
            lab2();
        else if (arg == "-l3")
            lab3();
    }

    return 0;
}
