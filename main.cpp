#include "imageutils.h"
#include <iostream>

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

int main()
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

    return 0;
}
