/**
 * @file parameter_adjustment.cpp
 * @brief 图像预处理参数实时调节工具
 *
 * 本程序使用 OpenCV 实现了一个带有滑动条控制面板的图像处理演示工具。
 * 用户可以通过调整滑动条实时改变中值滤波、高斯滤波、Canny 边缘检测
 * 以及形态学闭/开运算的参数，并立即看到处理后的图像效果。
 * 适用于寻找最佳预处理参数组合，或教学演示图像处理流程。
 */

#include <opencv2/imgcodecs.hpp>   
#include <opencv2/highgui.hpp>     
#include <opencv2/imgproc.hpp>     
#include <iostream>                
#include <string>                  

using namespace std;
using namespace cv;

// ========================= 全局变量 =========================
// 用于存储当前滑动条的值，方便在回调函数中访问

int medianKsize = 3;      // 中值滤波核大小（必须为奇数，范围 1~31）
int gaussKsize = 5;       // 高斯滤波核大小（必须为奇数，范围 1~31）
int gaussSigma = 0;       // 高斯滤波 sigma 值（0 表示由 OpenCV 自动计算）
int cannyLow = 35;        // Canny 边缘检测低阈值
int cannyHigh = 484;      // Canny 边缘检测高阈值
int morphClose = 7;       // 闭运算结构元素大小（正方形边长，奇数）
int morphOpen = 3;        // 开运算结构元素大小（正方形边长，奇数）

Mat originalImage;        // 存储原始图像（程序启动时加载，处理时不会修改）
string controlWindow = "Controls";   // 控制面板窗口名称
string imageWindow = "Image";        // 结果显示窗口名称

/**
 * @brief 滑动条回调函数
 * 每当用户移动任意一个滑动条时，该函数都会被调用。
 * 它会按照固定的图像处理流水线，使用当前滑动条的值对原始图像进行处理，
 * 并将结果显示在 "Image" 窗口中，实现实时交互。
 */
void onTrackbarChange(int, void*) {
    if (originalImage.empty()) return;   // 确保原始图像已加载

    Mat processed;        // 最终处理结果
    Mat temp;             // 中间处理结果

    // ---------- 步骤1：中值滤波 ----------
    // 中值滤波能有效去除椒盐噪声，同时保留边缘。核大小必须是奇数。
    int medianSize = max(1, medianKsize);
    if (medianSize % 2 == 0) medianSize++;   // 强制转为奇数
    medianBlur(originalImage, temp, medianSize);

    // ---------- 步骤2：高斯滤波 ----------
    // 高斯滤波用于平滑图像、抑制高斯噪声，核大小和 sigma 影响平滑程度。
    int gaussSize = max(1, gaussKsize);
    if (gaussSize % 2 == 0) gaussSize++;     // 强制转为奇数
    double sigma = (gaussSigma <= 0) ? 0 : (double)gaussSigma;   // sigma <= 0 表示自动计算
    GaussianBlur(temp, temp, Size(gaussSize, gaussSize), sigma, sigma);

    // ---------- 步骤3：Canny 边缘检测 ----------
    // 低阈值用于边缘连接，高阈值用于检测强边缘。这里自动确保低 <= 高。
    int low = min(cannyLow, cannyHigh);
    int high = max(cannyLow, cannyHigh);
    Canny(temp, processed, low, high);

    // ---------- 步骤4：形态学处理 ----------
    // 闭运算（膨胀后腐蚀）可以填充边缘内部的细小孔洞，连接邻近的断裂边缘。
    // 开运算（腐蚀后膨胀）可以去除孤立的噪点边缘。
    int cSize = max(1, morphClose);
    if (cSize % 2 == 0) cSize++;
    int oSize = max(1, morphOpen);
    if (oSize % 2 == 0) oSize++;

    // 创建矩形结构元素（正方形核）
    Mat closeKernel = getStructuringElement(MORPH_RECT, Size(cSize, cSize));
    Mat openKernel = getStructuringElement(MORPH_RECT, Size(oSize, oSize));

    // 闭运算：膨胀 -> 腐蚀
    dilate(processed, processed, closeKernel, Point(-1, -1), 1);
    erode(processed, processed, closeKernel, Point(-1, -1), 1);

    // 开运算：腐蚀 -> 膨胀
    erode(processed, processed, openKernel, Point(-1, -1), 1);
    dilate(processed, processed, openKernel, Point(-1, -1), 1);

    // 在图像窗口中显示最终结果
    imshow(imageWindow, processed);
}

/**
 * @brief 主函数：初始化窗口、创建滑动条、进入事件循环
 * @return 程序执行状态（0 成功，-1 图像加载失败）
 */
int main() {
    // ---------- 加载原始图像 ----------
    // 请根据实际路径修改文件名，建议使用相对路径或绝对路径
    originalImage = imread("object/110RAY2.jpg");
    if (originalImage.empty()) {
        cerr << "错误：无法加载图像，请检查路径 " << "object/110RAY2.jpg" << endl;
        return -1;
    }

    // ---------- 创建窗口 ----------
    // 控制面板窗口：用于放置所有滑动条
    namedWindow(controlWindow, WINDOW_NORMAL);
    resizeWindow(controlWindow, 400, 300);

    namedWindow(imageWindow, WINDOW_NORMAL);
    resizeWindow(imageWindow, originalImage.cols / 2, originalImage.rows / 2);

    // ---------- 添加滑动条 ----------
    // 每个滑动条都绑定到对应的全局变量，范围根据实际需要设置
    createTrackbar("Median Ksize", controlWindow, &medianKsize, 31, onTrackbarChange);
    createTrackbar("Gauss Ksize", controlWindow, &gaussKsize, 31, onTrackbarChange);
    createTrackbar("Gauss Sigma", controlWindow, &gaussSigma, 50, onTrackbarChange);
    createTrackbar("Canny Low", controlWindow, &cannyLow, 500, onTrackbarChange);
    createTrackbar("Canny High", controlWindow, &cannyHigh, 500, onTrackbarChange);
    createTrackbar("Close Size", controlWindow, &morphClose, 31, onTrackbarChange);
    createTrackbar("Open Size", controlWindow, &morphOpen, 31, onTrackbarChange);

    onTrackbarChange(0, nullptr);

    // ---------- 主事件循环 ----------
    cout << "调整控制窗口中的滑动条以实时查看效果，按 ESC 退出。" << endl;
    while (true) {
        int key = waitKey(30);   
        if (key == 27)           
            break;
    }

    // 退出前关闭所有 OpenCV 创建的窗口
    destroyAllWindows();
    return 0;
}