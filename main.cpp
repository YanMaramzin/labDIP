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
    Mat blur3;
    Mat blur32;
    Mat blur5;
    Mat blur52;
    ImageUtils::gauss(imageMono, blur3, 3, 0.6);
    ImageUtils::gauss(imageMono, blur32, 3, 0.8);
    ImageUtils::gauss(imageMono, blur5, 5, 0.6);
    ImageUtils::gauss(imageMono, blur52, 5, 0.8);
    cv::imshow("До", imageMono);
    cv::imshow("Фильтр Гаусса апертура 3", blur3);
    cv::imshow("Фильтр Гаусса апертура 5", blur5);

    // Фильтр мозаика
    Mat mosaic1;
    ImageUtils::mosaic(imageMono, mosaic1, 3);
    cv::imshow("Фильтр мозаика", mosaic1);

    // Апертурная коррекция
    Mat aper;
    ImageUtils::apertureCorrection(imageMono, aper, 30);
    cv::imshow("Апертурная коррекция", aper);

    // Медианная фильтрация
    cv::Mat noise(imageMono.size(), imageMono.type());
    float m = (10, 12, 34);
    float sigma = (1, 5, 50);
    cv::randn(noise, m, sigma);
    auto imageMonoNoise = imageMono.clone();
    imageMonoNoise += noise;
    cv::imshow("Шум", imageMonoNoise);
    Mat med;
    ImageUtils::medianFilter(imageMonoNoise, med);
    cv::imshow("Медианная фильтрация", med);

    // Разность гауссианов
    Mat diff;
    cv::absdiff(blur52, blur3, diff);
    cv::imshow("Разность гауссианов;", diff * 15);

    // Контур Кэнни
    Mat contour;
    cv::Canny(image, contour, 50, 200);
    cv::imshow("Контур Кэнни", contour);


    // Робертс горизонтальные контуры
    Mat robertX;
    Mat kernel_x = (cv::Mat_<int>(2, 2) << 1, 0, 0, -1);
    filter2D(imageMono, robertX, -1, kernel_x, cv::Point(-1, -1), 0.0);
    cv::imshow("Robert_x", robertX);

    // Робертс вертикальные контуры
    Mat robertY;
    Mat kernel_Y = (cv::Mat_<int>(2, 2) << 0, 1, -1, 0);
    filter2D(imageMono, robertY, -1, kernel_Y, cv::Point(-1, -1), 0.0);
    cv::imshow("Robert_y", robertY);

    cv::waitKey();
}

void lab4()
{
    auto image = cv::imread("/home/mist/Downloads/zoro.jpeg");
    auto imageMono = ImageUtils::monochromeImage(image);
    auto imageBinary = ImageUtils::quantizedImage(imageMono, 2);
    Mat imageOutput;
    Mat imageDil;
    Mat imageOpen;
    Mat imageClose;
    Mat imageMonoEros;
    Mat imageMonoDil;
    Mat countur;
    ImageUtils::erosion(imageBinary, imageOutput);
    ImageUtils::dilation(imageBinary, imageDil);
    Mat open = ImageUtils::opening(imageBinary, imageOpen);
    Mat close = ImageUtils::close(imageBinary, imageClose);
    ImageUtils::countur(imageBinary, countur);
    ImageUtils::erosion(imageMono, imageMonoEros);
    ImageUtils::dilation(imageMono, imageMonoDil);
    Mat openMono = ImageUtils::opening(imageMono, imageOpen);
    Mat closeMono = ImageUtils::close(imageMono, imageClose);
    cv::imshow("Исходное", imageBinary);
    cv::imshow("Эрозия", imageOutput);
    cv::imshow("Дилатация", imageDil);
    cv::imshow("Открытие", open);
    cv::imshow("Закрытие", close);
    cv::imshow("Контур", countur);

    cv::imshow("Исходное полутовное", imageMono);
    cv::imshow("Эрозия полутовное", imageMonoEros);
    cv::imshow("Дилатация полутовное", imageMonoDil);
    cv::imshow("Открытие полутовное", openMono);
    cv::imshow("Закрытие полутовное", closeMono);

    cv::waitKey();
    std::cout << "Пока в процессе";
}

void lab5()
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
        else if (arg == "-l5")
            lab5();
    }

    return 0;
}
