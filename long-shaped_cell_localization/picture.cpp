/**
 * @file picture.cpp
 * @brief 单细胞图像分割与角点检测主程序
 *
 * 本程序读取一张显微图像，依次执行：
 * 1. 预处理（中值滤波、高斯滤波、Canny边缘检测、形态学闭/开运算）
 * 2. 轮廓检测与细胞图像分割（基于面积筛选，自动裁剪单个细胞区域）
 * 3. 对每个分割出的细胞图像进行 Harris 角点检测，并绘制两个最远角点（标记为头部和尾部）
 * 4. 保存结果图像并输出总耗时
 */

#include <opencv2/imgcodecs.hpp>   
#include <opencv2/highgui.hpp>     
#include <opencv2/imgproc.hpp>     
#include <opencv2/core/core.hpp>   
#include <iostream>                
#include "imgFeat.h"               
#include <chrono>                 
#include <string>                  

using namespace std;
using namespace cv;
using namespace chrono;

// 全局变量：存储每个细胞的外接矩形（预分配 50 个，若超过 50 会越界）
vector<Rect> boundRect(50);

/**
 * @brief 计算两点之间的欧氏距离
 * @param p1 第一个点
 * @param p2 第二个点
 * @return 距离值
 */
float calculateDistance(Point p1, Point p2)
{
    return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
}

/**
 * @brief 从二值边缘图中提取轮廓，筛选面积大于 1500 的区域，分割出单个细胞图像
 * @param imgero 输入的二值图像（形态学处理后的边缘图）
 * @param img    原始灰度/彩色图像（用于提取像素）
 * @param cellImages 输出参数，存储每个分割出的细胞图像（Mat 对象）
 *
 * 处理流程：
 * - 使用 findContours 提取外轮廓
 * - 对每个轮廓计算面积，若 >=1500 则计算其外接矩形并向外扩展 12 像素
 * - 创建掩膜，仅保留当前轮廓区域，提取该区域子图并存入 cellImages
 */
void getContours(Mat imgero, Mat img, vector<Mat>& cellImages)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(imgero, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 将单通道二值图转换为三通道灰度图（用于显示彩色）
    vector<Mat> channels;
    Mat imgColor;
    channels.push_back(img); channels.push_back(img); channels.push_back(img);
    merge(channels, imgColor);
    Mat imgsave = imgColor;   // 备份原始彩色图像，用于恢复

    int no = 0, k = 0;
    for (int i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        string objectno = "microcell";
        if (area >= 1500)   // 面积阈值，筛选出细胞区域
        {
            boundRect[k] = boundingRect(contours[i]);
            int padding = 12;
            // 扩展矩形并限制在图像边界内
            boundRect[k].x = max(boundRect[k].x - padding, 0);
            boundRect[k].y = max(boundRect[k].y - padding, 0);
            boundRect[k].width = min(boundRect[k].width + 2 * padding, img.cols - boundRect[k].x);
            boundRect[k].height = min(boundRect[k].height + 2 * padding, img.rows - boundRect[k].y);

            if (boundRect[k].x > 0 && boundRect[k].y > 0 &&
                boundRect[k].x + boundRect[k].width < img.cols &&
                boundRect[k].y + boundRect[k].height < img.rows)
            {
                objectno = objectno + to_string(no);
                no++;

                // 创建掩膜，仅绘制当前轮廓
                Mat mask = Mat::zeros(img.size(), CV_8UC1);
                drawContours(mask, contours, i, Scalar(255), FILLED);
                Mat result;
                img.copyTo(result, mask);

                // 将当前细胞区域（带掩膜）保存到 cellImages
                cellImages.push_back(result(boundRect[k]).clone());

                // 注意：此处修改了 img，但随后又恢复，可能导致后续轮廓受影响（设计问题）
                img = imgsave;
                k++;
            }
        }
    }
}


/**
 * @brief Harris 角点检测
 * @param imgSrc 输入图像（彩色或灰度）
 * @param imgDst 输出二值图像，角点处像素为 255
 * @param alpha  Harris 响应函数中的自由参数
 *
 * 实现细节：计算图像梯度 Ix, Iy，构建 M 矩阵，计算 R = det(M) - α·trace(M)^2，
 * 然后进行非极大值抑制和阈值筛选。
 */
void feat::detectHarrisCorners(const Mat& imgSrc, Mat& imgDst, double alpha)
{
    Mat gray;
    if (imgSrc.channels() == 3)
        cvtColor(imgSrc, gray, COLOR_BGR2GRAY);
    else
        gray = imgSrc.clone();
    gray.convertTo(gray, CV_64F);

    // 使用 Sobel 算子近似（这里用简单的 [-1,0,1] 差分核）
    Mat xKernel = (Mat_<double>(1, 3) << -1, 0, 1);
    Mat yKernel = xKernel.t();
    Mat Ix, Iy;
    filter2D(gray, Ix, CV_64F, xKernel);
    filter2D(gray, Iy, CV_64F, yKernel);

    // 计算 Ixx, Iyy, Ixy
    Mat Ix2, Iy2, Ixy;
    Ix2 = Ix.mul(Ix);
    Iy2 = Iy.mul(Iy);
    Ixy = Ix.mul(Iy);

    // 高斯平滑（窗口大小 7，sigma=1）
    Mat gaussKernel = getGaussianKernel(7, 1);
    filter2D(Ix2, Ix2, CV_64F, gaussKernel);
    filter2D(Iy2, Iy2, CV_64F, gaussKernel);
    filter2D(Ixy, Ixy, CV_64F, gaussKernel);

    // 计算每个像素的 Harris 响应值
    Mat cornerStrength(gray.size(), gray.type());
    for (int i = 0; i < gray.rows; i++)
    {
        for (int j = 0; j < gray.cols; j++)
        {
            double det_m = Ix2.at<double>(i, j) * Iy2.at<double>(i, j) - Ixy.at<double>(i, j) * Ixy.at<double>(i, j);
            double trace_m = Ix2.at<double>(i, j) + Iy2.at<double>(i, j);
            cornerStrength.at<double>(i, j) = det_m - alpha * trace_m * trace_m;
        }
    }

    // 阈值化 + 非极大值抑制
    double maxStrength;
    minMaxLoc(cornerStrength, NULL, &maxStrength, NULL, NULL);
    Mat dilated;
    Mat localMax;
    dilate(cornerStrength, dilated, Mat());
    compare(cornerStrength, dilated, localMax, CMP_EQ);

    Mat cornerMap;
    double qualityLevel = 0.01;
    double thresh = qualityLevel * maxStrength;
    cornerMap = cornerStrength > thresh;
    bitwise_and(cornerMap, localMax, cornerMap);
    imgDst = cornerMap.clone();
}

/**
 * @brief 在图像上绘制角点（简单版本，每个角点画一个绿色圆）
 * @param image 待绘制的彩色图像（会被修改）
 * @param binary 角点二值掩膜
 */
void feat::drawCornerOnImage(Mat& image, const Mat& binary)
{
    Mat_<uchar>::const_iterator it = binary.begin<uchar>();
    Mat_<uchar>::const_iterator itd = binary.end<uchar>();
    for (int i = 0; it != itd; ++it, ++i)
    {
        if (*it)
            circle(image, Point(i % image.cols, i / image.cols), 3, Scalar(0, 255, 0), 1);
    }
}

/**
 * @brief 在图像上绘制角点，并找出距离最远的两个角点（标记为头部和尾部）
 * @param image 待绘制的彩色图像（会被修改）
 * @param binary 角点二值掩膜
 * @param collect 用于临时存储角点坐标的 vector（需预分配足够大小）
 * @param m 当前细胞索引，用于获取全局 boundRect 偏移量（用于坐标还原）
 *
 * 说明：该函数会将所有角点坐标存入 collect，然后找到与第一个角点距离最远的点作为尾部，
 *       再找到与第一个角点距离次远的点作为头部（逻辑有些特殊，但保持了原有实现）。
 *       最后在图像上绘制这两个特殊角点（红色实心圆），并标注坐标文本。
 */
void feat::drawCornerOnImage(Mat& image, const Mat& binary, vector<Point2f>& collect, int m)
{
    Mat_<uchar>::const_iterator it = binary.begin<uchar>();
    Mat_<uchar>::const_iterator itd = binary.end<uchar>();
    int j = 0, MaxNum = 0;
    float distMax = 0, xMax = 0, yMax = 0;

    // 收集所有角点坐标（注意 collect 必须足够大，否则越界）
    for (int i = 0; it != itd; ++it, ++i)
    {
        if (*it)
        {
            collect[j] = Point2f(i % image.cols, i / image.cols);
            cout << collect[j].x << "," << collect[j].y << endl;
            j++;
        }
    }

    // 以下算法寻找两个最远角点：先找与 collect[1] 最远的点，交换到 collect[2]；
    // 再找与 collect[1] 次远的点（实际仍是距离最大，逻辑上应找与 collect[1] 最远的两个不同点）
    int i = 0;
    for (i = 1; i < j; i++)
    {
        float dist = (collect[i].x - collect[1].x) * (collect[i].x - collect[1].x) +
            (collect[i].y - collect[1].y) * (collect[i].y - collect[1].y);
        if (dist > distMax)
        {
            distMax = dist;
            MaxNum = i;
        }
    }
    Point2f temp = collect[MaxNum];
    collect[MaxNum] = collect[1];
    collect[1] = temp;

    distMax = 0;
    for (i = 1; i < j; i++)
    {
        float dist = (collect[i].x - collect[1].x) * (collect[i].x - collect[1].x) +
            (collect[i].y - collect[1].y) * (collect[i].y - collect[1].y);
        if (dist > distMax)
        {
            distMax = dist;
            MaxNum = i;
        }
    }
    temp = collect[MaxNum];
    collect[MaxNum] = collect[2];
    collect[2] = temp;

    // 在图像上绘制头部和尾部（红色圆点）
    circle(image, collect[1], 4, Scalar(0, 0, 255), -1);
    circle(image, collect[2], 4, Scalar(0, 0, 255), -1);

    // 计算在原图（未裁剪前）中的绝对坐标（boundRect[m] 存储了细胞区域在原图中的位置）
    string strHead = "head:(" + to_string(int(collect[1].x + boundRect[m].x - 12)) + "," +
        to_string(int(collect[1].y + boundRect[m].y - 12)) + ")";
    string strTail = "tail:(" + to_string(int(collect[2].x + boundRect[m].x - 12)) + "," +
        to_string(int(collect[2].y + boundRect[m].y - 12)) + ")";
    putText(image, strHead, Point(0, collect[1].y / 2 + collect[2].y / 2),
        FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 165, 255));
    putText(image, strTail, Point(0, collect[1].y / 2 + collect[2].y / 2 + 25),
        FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 165, 255));
}

// ==================== 主函数 ====================
int main()
{
    // 读取原始图像（路径可根据需要修改）
    Mat img = imread("object/110RAY2.jpg");

    // 开始计时
    auto beforeTime = std::chrono::steady_clock::now();

    Mat imgmedianblur, imggaussianblur, imgoutput, imgotsu, imgdilate, imgero, imgcanny;

    // ---------- 预处理阶段 ----------
    // 1. 中值滤波（核大小 3）
    medianBlur(img, imgmedianblur, 3);
    // 2. 高斯滤波（核大小 5x5，sigma 自动）
    GaussianBlur(imgmedianblur, imggaussianblur, Size(5, 5), 0, 0);
    // 3. Canny 边缘检测（阈值 35 和 484）
    Canny(imggaussianblur, imgcanny, 35, 484);
    // 4. 形态学处理：先闭运算（填充孔洞），再开运算（去除噪点）
    Mat closeKernel = getStructuringElement(MORPH_RECT, Size(7, 7));
    Mat openKernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(imgcanny, imgdilate, closeKernel);
    erode(imgdilate, imgero, closeKernel);
    erode(imgero, imgero, openKernel);
    dilate(imgero, imgdilate, openKernel);

    // 显示处理后的二值边缘图像（窗口保持宽高比）
    cv::namedWindow("microcellero", 0x00000010);  // 0x00000010 = WINDOW_KEEPRATIO
    imshow("microcellero", imgdilate);

    // ---------- 细胞分割 ----------
    vector<Mat> cellImages;   // 存储每个分割出的单细胞图像（灰度/二值）
    getContours(imgdilate, imgdilate, cellImages);

    // 修复第一个细胞图像因灰度显示问题导致的颜色异常（强制转为三通道）
    vector<Mat> channels;
    Mat imgColor;
    channels.push_back(cellImages[0]);
    channels.push_back(cellImages[0]);
    channels.push_back(cellImages[0]);
    merge(channels, imgColor);
    vector<Mat> imgshow = cellImages;   // imgshow 用于显示彩色版本
    imgshow[0] = imgColor;

    int cellimgno = cellImages.size();
    vector<Point2f> collect(1000);      // 预分配足够大的角点存储空间

    // ---------- 对每个细胞进行角点检测并输出 ----------
    for (size_t i = 0; i < cellimgno; i++)
    {
        // 检测 Harris 角点，结果存入 cellImages[i]（二值掩膜）
        feat::detectHarrisCorners(imgshow[i], cellImages[i], 0.120);
        // 在彩色图像上绘制角点，并标记最远两点
        feat::drawCornerOnImage(imgshow[i], cellImages[i], collect, i);
        // 显示单个细胞图像
        imshow("singlecell" + to_string(i), imgshow[i]);
        // 保存到 output 文件夹
        imwrite("output/singlecell" + to_string(i) + ".jpg", imgshow[i]);
    }

    // 结束计时并输出总耗时（微秒）
    auto afterTime = std::chrono::steady_clock::now();
    double duration_microsecond = std::chrono::duration<double, std::micro>(afterTime - beforeTime).count();
    std::cout << "总耗时:" << duration_microsecond << "微秒" << std::endl;

    waitKey(0);
    return 0;
}