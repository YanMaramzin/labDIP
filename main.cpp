#include "imageutils.h"

int main()
{
    auto width = 300;
    auto heigth = 300;
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
    return 0;
}
