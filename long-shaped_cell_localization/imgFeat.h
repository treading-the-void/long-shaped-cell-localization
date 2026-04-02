/**
 * @file imgFeat.h
 * @brief 图像特征提取模块头文件
 *
 * 本模块提供 Harris 角点检测及相关绘制功能。
 * 声明了 feat 命名空间下的角点检测和绘制函数。
 */

#ifndef IMGFEAT_H_
#define IMGFEAT_H_

#include <iostream>
#include <algorithm>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

namespace feat
{
    // 极小常量，用于浮点数比较（保留以备后续扩展）
    const double EPS = 2.2204e-16;

    /**
     * @brief Harris 角点检测
     * @param imgSrc 输入图像（彩色或灰度）
     * @param imgDst 输出二值图像，角点处像素值为 255
     * @param alpha  Harris 响应函数中的自由参数（通常 0.04~0.06）
     *
     * 该函数计算每个像素的 Harris 响应值，经过非极大值抑制和阈值筛选后，
     * 将角点位置以二值图像形式输出。
     */
    void detectHarrisCorners(const Mat& imgSrc, Mat& imgDst, double alpha);

    /**
     * @brief 在彩色图像上绘制角点（简单版本）
     * @param image 待绘制的彩色图像（会被修改）
     * @param binary 角点二值掩膜（每个角点位置为 255）
     *
     * 在 binary 中每个非零点处绘制一个半径为 3 的绿色圆点。
     */
    void drawCornerOnImage(Mat& image, const Mat& binary);

    /**
     * @brief 在彩色图像上绘制角点，并标记最远两点（头部和尾部）
     * @param image 待绘制的彩色图像（会被修改）
     * @param binary 角点二值掩膜
     * @param collect 临时存储角点坐标的 vector（需预分配足够空间）
     * @param m      当前细胞索引，用于获取全局偏移量（配合 boundRect 使用）
     *
     * 该函数会找出所有角点中距离最远的两个点，用红色实心圆标记，
     * 并在图像上显示其坐标文本。
     */
    void drawCornerOnImage(Mat& image, const Mat& binary, vector<Point2f>& collect, int m);

} // namespace feat

#endif // IMGFEAT_H_