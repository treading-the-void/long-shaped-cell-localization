/**
 * @file video.cpp
 * @brief 视频实时处理与细胞检测工具
 *
 * 本程序读取一个视频文件，对每一帧执行以下操作：
 * 1. 高斯滤波（平滑去噪）
 * 2. Canny 边缘检测
 * 3. 形态学膨胀与腐蚀（闭运算，填充边缘断裂）
 * 4. 轮廓检测，根据面积筛选细胞区域，并绘制外接矩形框
 * 5. 将处理结果（带矩形框的图像）缩放后显示并写入输出视频文件
 *
 * 可以通过调整采样率（RATE）来跳帧处理，提高处理速度。
 */

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat img;               // 全局变量，用于存储绘制矩形框后的图像（在 getContours 中更新）
#define RATE 8         // 采样率：每 RATE 帧处理一帧（跳帧处理，提高速度）

/**
 * @brief 在二值边缘图像上检测轮廓，并筛选出面积大于阈值的区域，绘制外接矩形
 * @param imgero 输入的二值边缘图像（经过形态学处理后的 Canny 结果）
 *
 * 处理流程：
 * - 使用 findContours 提取所有外轮廓
 * - 对每个轮廓计算面积，若 >=1000 则视为细胞区域
 * - 计算该轮廓的外接矩形，并向外扩展 12 像素（便于观察）
 * - 在全局图像 img（三通道彩色）上绘制矩形框（洋红色，线宽 3）
 *
 * 注意：全局变量 img 需要提前初始化为与输入图像同尺寸的彩色图像。
 */
void getContours(Mat imgero)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(imgero, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 将单通道二值图像转换为三通道彩色图像，以便显示彩色矩形
    vector<Mat> channels;
    Mat imgColor;
    channels.push_back(imgero);
    channels.push_back(imgero);
    channels.push_back(imgero);
    merge(channels, imgColor);
    img = imgColor;   // 更新全局显示图像

    int k = 0;
    vector<Rect> boundRect(contours.size());   // 存储满足条件的轮廓外接矩形
    for (int i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area >= 1000)   // 面积阈值，筛选细胞区域（可根据实际调整）
        {
            boundRect[k] = boundingRect(contours[i]);
            int padding = 12;   // 矩形向外扩展的像素数，使框更宽松
            boundRect[k].x = max(boundRect[k].x - padding, 0);
            boundRect[k].y = max(boundRect[k].y - padding, 0);
            boundRect[k].width = min(boundRect[k].width + 2 * padding, imgero.cols - boundRect[k].x);
            boundRect[k].height = min(boundRect[k].height + 2 * padding, imgero.rows - boundRect[k].y);

            // 确保矩形完全在图像内部
            if (boundRect[k].x > 0 && boundRect[k].y > 0 &&
                boundRect[k].x + boundRect[k].width < imgero.cols &&
                boundRect[k].y + boundRect[k].height < imgero.rows)
            {
                // 在全局图像上绘制洋红色矩形框
                rectangle(img, Point2f(boundRect[k].x, boundRect[k].y),
                    Point2f(boundRect[k].x + boundRect[k].width, boundRect[k].y + boundRect[k].height),
                    Scalar(255, 0, 255), 3);
                k++;
            }
        }
    }
}

/**
 * @brief 主函数：打开视频，逐帧处理并输出结果视频
 * @return 0 表示成功
 */
int main()
{
    string str;
    VideoCapture capture;
    capture.open("object/Ray 1.mp4");   // 视频文件路径，请根据实际情况修改

    if (!capture.isOpened())
    {
        cout << "fail to open video file!" << endl;
        return -1;
    }

    // 获取视频基本信息
    long totalFrameNumber = capture.get(CAP_PROP_FRAME_COUNT);
    cout << "This video has : " << totalFrameNumber << " frames" << endl;

    double rate = capture.get(CAP_PROP_FPS);
    cout << "The frame rate is : " << rate << endl;

    int width = static_cast<int>(capture.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(capture.get(CAP_PROP_FRAME_HEIGHT));
    cout << "The image resolution is : " << width << " x " << height << endl;

    clock_t start, end;   // 用于计时

    // 准备输出视频（mp4 格式，帧率为原始视频帧率除以采样率，尺寸与原视频相同）
    VideoWriter OutputVideo("./output/video_output.mp4",
        VideoWriter::fourcc('m', 'p', '4', 'v'),
        double(rate / RATE),
        cv::Size(width, height));

    Mat frame;                     // 当前帧
    Mat imgmedianblur, imggaussianblur, imgbil, imgcanny, imgdilate, imgero, dst, src;
    int count = 0;                 // 当前帧计数
    start = clock();

    while (1)
    {
        count++;
        capture >> frame;
        if (frame.empty())
            break;

        cout << count << "-" << totalFrameNumber << '\r';   // 显示进度

        // 跳帧处理：只处理每 RATE 帧中的一帧，其余帧跳过（提高速度）
        if (count % RATE != 0)
            continue;

        // ---------- 图像预处理 ----------
        // 高斯滤波（核大小 3x3，sigma 自动）
        GaussianBlur(frame, imggaussianblur, Size(3, 3), 0, 0);

        // Canny 边缘检测（低阈值 35，高阈值 420）
        Canny(imggaussianblur, imgcanny, 35, 420);

        // 形态学处理：膨胀 + 腐蚀（相当于闭运算，填充边缘断裂）
        Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));
        dilate(imgcanny, imgdilate, kernel);   // 膨胀
        erode(imgdilate, imgero, kernel);      // 腐蚀

        // 调用轮廓检测与绘制函数（结果存储在全局 img 中）
        getContours(imgero);

        // 可选：在二值图像上绘制一个测试矩形（调试用）
        rectangle(imgero, Point2f(100, 100), Point2f(200, 200), Scalar(255, 0, 255), 5);

        // 缩放图像以便显示（缩小为原来的一半）
        resize(img, dst, Size(0, 0), 0.5, 0.5, INTER_AREA);
        resize(frame, src, Size(0, 0), 0.5, 0.5, INTER_AREA);

        // 显示原始帧（缩放后）和处理结果（带矩形框的彩色图）
        imshow("rawpic", src);
        moveWindow("rawpic", 400, 0);
        imshow("microcellero", dst);
        moveWindow("microcellero", 400, 300);

        // 将处理后的图像（带矩形框）写入输出视频
        OutputVideo.write(img);

        waitKey(1);   // 等待 1 毫秒，控制帧率（实际播放速度由视频写入控制）
    }

    end = clock();
    capture.release();
    OutputVideo.release();

    cout << endl;
    cout << "The program takes : " << end - start << " ms" << endl;
    system("pause");
    return 0;
}