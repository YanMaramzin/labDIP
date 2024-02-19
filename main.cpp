#include "imageutils.h"

int main()
{
    auto image = cv::imread("/home/mist/Downloads/zoro.jpeg");
    auto imageMono = ImageUtils::monochromeImage(image);
    // cv::imshow("Зоро", image);
    // cv::imwrite("/home/mist/zoro_in_Piter.jpg", imageMono);
    auto histImg = ImageUtils::hist(imageMono);;
    imshow("Зоро в Питере", imageMono);
    namedWindow("Гистограмма", cv::WINDOW_NORMAL);
    cv::resizeWindow("Гистограмма", 1000, 1000);
    imshow("Гистограмма", histImg);
    cv::waitKey();
    return 0;
}
