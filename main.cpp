#include <iostream>
#include "imageutils.h"

int blockSize = 8;

static void onMouse(int event, int x, int y, int, void *param)
{
    auto img = static_cast<Mat*>(param);
    if (event != cv::EVENT_LBUTTONDOWN)
        return;

    int blockX = x - x % blockSize;
    int blockY = y - y % blockSize;

    cv::Rect block2 = cv::Rect(blockX, blockY, blockSize, blockSize);
    Mat imageBlock = cv::Mat(*img, block2);
    auto DCT = ImageUtils::discretCosineTransform(ImageUtils::monochromeImage(imageBlock));
    cv::imshow("DCT", DCT);

    auto hist = ImageUtils::hist(DCT);
    cv::imshow("Гистограмма", hist);
    cv::imshow("Выбранный блок", imageBlock);
    cv::Rect block = cv::Rect(blockX, blockY, blockSize, blockSize);
    rectangle(*img, block, cv::Scalar(1, 255, 255));
    cv::imshow("Сетка", *img);

    cv::imwrite("блок.jpg", imageBlock);
    cv::imwrite("DCT.jpg", DCT);
    cv::imwrite("hist.jpg", hist);
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
    auto quantLevel = 64;
    auto quant = ImageUtils::quantizedImage(imageMono, quantLevel);
    auto intervalQuant = 255.0 / (quantLevel - 1);
    auto sko = ImageUtils::sko(imageMono, quant);
    auto skoOt = intervalQuant / sqrt(12);
    std::cout << sko  << " " << skoOt;
    auto quantHist = ImageUtils::hist(quant);
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
    cv::imwrite("Полутон.jpg", imageMono);
    Mat imageWithBlocks = image.clone();

    for (int row = 0; row < imageMono.rows; row = row + blockSize) {
        for (int col = 0; col < imageMono.cols; col = col + blockSize) {
            cv::Rect block = cv::Rect(col, row, blockSize, blockSize);
            rectangle(imageWithBlocks, block, cv::Scalar(0, 0, 0));
        }
    }

    cv::imwrite("Сетка.jpg", imageWithBlocks);
    cv::imshow("Сетка", imageWithBlocks);
    cv::setMouseCallback("Сетка", onMouse, &image);

    cv::waitKey();
}

void lab3()
{
    auto image = cv::imread("/home/mist/Downloads/zoro.jpeg");
    auto imageMono = ImageUtils::monochromeImage(image);
//    Mat blur3;
//    Mat blur32;
//    Mat blur5;
//    Mat blur52;
//    ImageUtils::gauss(imageMono, blur3, 3, 0.6);
//    ImageUtils::gauss(imageMono, blur32, 3, 0.8);
//    ImageUtils::gauss(imageMono, blur5, 5, 0.6);
//    ImageUtils::gauss(imageMono, blur52, 5, 0.8);
//    cv::imshow("До", imageMono);
//    cv::imshow("После 3", blur3);
//    cv::imshow("После 5", blur5);
//    Mat diff;
//    cv::absdiff(blur52, blur3, diff);
//    cv::imshow("Контур", diff * 15);
//    Mat contour;
//    cv::Canny(image, contour, 50, 200);
//    cv::imshow("Контур Кэнни", contour);

    // Медианный фильтр
//    Mat mosaic1;
//    ImageUtils::mosaic(imageMono, mosaic1, 3);
//    cv::imshow("", mosaic1);

    // Апертурная коррекция
    Mat aper;
    ImageUtils::apertureCorrection(imageMono, aper, 40);
    cv::imshow("Апертруная коррекция", aper);

    Mat med;
    ImageUtils::medianFilter(imageMono, med);
    cv::imshow("Медианная фильтрация", med);

    cv::medianBlur(imageMono, med, 3);
    cv::imshow("Медианная фильтрацияCV", med);

    // Robert X direction
    Mat robertX;
    Mat kernel_x = (cv::Mat_<int>(2, 2) << 1, 0, 0, -1);
    filter2D(imageMono, robertX, -1, kernel_x, cv::Point(-1, -1), 0.0);
    // - 1 is the depth of the image, the program automatically determines -1
    cv::imshow("Robert_x", robertX);

    // Robert Y direction
    Mat robertY;
    Mat kernel_Y = (cv::Mat_<int>(2, 2) << 0, 1, -1, 0);
    filter2D(imageMono, robertY, -1, kernel_Y, cv::Point(-1, -1), 0.0);
    cv::imshow("Robert_y", robertY);

    cv::waitKey();
    std::cout << "Пока в процессе";
}

void lab4()
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
        else if (arg == "-l4")
            lab4();
    }

    return 0;
}
