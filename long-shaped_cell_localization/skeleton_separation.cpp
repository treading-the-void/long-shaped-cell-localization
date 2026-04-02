/**
 * @file skeleton_separation.cpp
 * @brief 基于骨架分析的粘连细胞分割工具
 *
 * 主要流程：
 * 1. 二值化
 * 2. 对每个连通域进行 Zhang-Suen 细化，得到骨架
 * 3. 构建骨架图，识别分支点（度数为 3 的点）
 * 4. 根据分支点判断是否分割，通过阻断骨架上的分支点来分离粘连区域
 * 5. 输出分割后的各个细胞掩膜，并可视化骨架和分割结果
 *
 * 适用场景：显微图像中粘连细胞的自动分离
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>

using namespace cv;
using namespace std;

// 8 邻域方向偏移量（用于连通性判断）
const int dx8[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
const int dy8[] = { -1, 0, 1, -1, 1, -1, 0, 1 };

// 自定义 Point 比较器，用于在 map/set 中作为 key
struct PointCompare {
    bool operator()(const Point& a, const Point& b) const {
        return (a.x < b.x) || (a.x == b.x && a.y < b.y);
    }
};

// ---------------------- 1. 二值化 ----------------------
/**
 * @brief 将输入图像转换为二值图像（前景白色，背景黑色）
 * @param src 输入图像（彩色或灰度均可）
 * @return 二值图像（CV_8UC1）
 *
 * 处理步骤：转灰度 -> 确保类型为 CV_8U -> Otsu 自动阈值
 */
Mat binarize(const Mat& src) {
    Mat gray, bin;

    // 1. 先转灰度（无论输入是什么）
    if (src.channels() == 3)
        cvtColor(src, gray, COLOR_BGR2GRAY);
    else if (src.channels() == 4)
        cvtColor(src, gray, COLOR_BGRA2GRAY);
    else if (src.channels() == 1)
        gray = src.clone();
    else {
        // 其他情况，取第一个通道
        vector<Mat> channels;
        split(src, channels);
        gray = channels[0];
    }

    // 2. 确保是 CV_8U 类型
    if (gray.type() != CV_8U) {
        gray.convertTo(gray, CV_8U);
    }

    // 3. 二值化（Otsu 自动阈值）
    threshold(gray, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

    return bin;
}

// ---------------------- 2. Zhang-Suen 细化算法 ----------------------
/**
 * @brief Zhang-Suen 并行细化算法，将二值图像细化为单像素宽的骨架
 * @param src 输入二值图像（白色前景）
 * @param dst 输出骨架图像（前景为骨架点）
 *
 * 算法原理：迭代删除满足特定条件的边界点，直到不再变化。
 * 分两个子迭代，分别处理不同的边界条件，以保持骨架连通性。
 */
void thinningZhangSuen(const Mat& src, Mat& dst) {
    dst = src.clone();
    vector<Point> del;
    int rows = dst.rows, cols = dst.cols;
    bool changed = true;
    while (changed) {
        changed = false;
        // Step 1
        for (int y = 1; y < rows - 1; ++y) {
            for (int x = 1; x < cols - 1; ++x) {
                if (dst.at<uchar>(y, x) == 0) continue;
                // 取 8 邻域像素值（0/1）
                int p2 = (dst.at<uchar>(y - 1, x) > 0) ? 1 : 0;
                int p3 = (dst.at<uchar>(y - 1, x + 1) > 0) ? 1 : 0;
                int p4 = (dst.at<uchar>(y, x + 1) > 0) ? 1 : 0;
                int p5 = (dst.at<uchar>(y + 1, x + 1) > 0) ? 1 : 0;
                int p6 = (dst.at<uchar>(y + 1, x) > 0) ? 1 : 0;
                int p7 = (dst.at<uchar>(y + 1, x - 1) > 0) ? 1 : 0;
                int p8 = (dst.at<uchar>(y, x - 1) > 0) ? 1 : 0;
                int p9 = (dst.at<uchar>(y - 1, x - 1) > 0) ? 1 : 0;
                // A：顺时针方向 0->1 的变换次数
                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                    (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                    (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                    (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                if (A == 1 && (B >= 2 && B <= 6)) {
                    int m1 = p2 * p4 * p6;
                    int m2 = p4 * p6 * p8;
                    if (m1 == 0 && m2 == 0)
                        del.push_back(Point(x, y));
                }
            }
        }
        for (auto& p : del) dst.at<uchar>(p.y, p.x) = 0;
        changed = !del.empty();
        del.clear();

        // Step 2
        for (int y = 1; y < rows - 1; ++y) {
            for (int x = 1; x < cols - 1; ++x) {
                if (dst.at<uchar>(y, x) == 0) continue;
                int p2 = (dst.at<uchar>(y - 1, x) > 0) ? 1 : 0;
                int p3 = (dst.at<uchar>(y - 1, x + 1) > 0) ? 1 : 0;
                int p4 = (dst.at<uchar>(y, x + 1) > 0) ? 1 : 0;
                int p5 = (dst.at<uchar>(y + 1, x + 1) > 0) ? 1 : 0;
                int p6 = (dst.at<uchar>(y + 1, x) > 0) ? 1 : 0;
                int p7 = (dst.at<uchar>(y + 1, x - 1) > 0) ? 1 : 0;
                int p8 = (dst.at<uchar>(y, x - 1) > 0) ? 1 : 0;
                int p9 = (dst.at<uchar>(y - 1, x - 1) > 0) ? 1 : 0;
                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                    (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                    (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                    (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                if (A == 1 && (B >= 2 && B <= 6)) {
                    int m1 = p2 * p4 * p8;
                    int m2 = p2 * p6 * p8;
                    if (m1 == 0 && m2 == 0)
                        del.push_back(Point(x, y));
                }
            }
        }
        for (auto& p : del) dst.at<uchar>(p.y, p.x) = 0;
        changed = changed || !del.empty();
        del.clear();
    }
}

// ---------------------- 3. 骨架图构建 ----------------------
/**
 * @brief 骨架图结构
 *
 * 将骨架点映射为图中的节点，根据 8 邻域连通性建立边。
 * 同时计算每个节点的“边缘度数”——通过分析固定半径方块边缘上的骨架点聚类数量，
 * 来判断该点是端点、普通点还是分支点。
 */
struct SkeletonGraph {
    map<Point, int, PointCompare> pointToId;  // 坐标 -> 节点ID
    vector<Point> points;                     // 节点坐标列表
    vector<set<int>> adj;                     // 邻接表（无向）
    vector<int> degree;                       // 节点度数（边缘聚类数）
    vector<bool> isJunction;                  // 是否为分支点（度数 >= 3）
    vector<bool> isEndpoint;                  // 是否为端点（度数 == 1）
};

/**
 * @brief 计算骨架中某点的“边缘度数”
 * @param skeleton 骨架图像
 * @param cx, cy 中心点坐标
 * @param radius 方块半径（默认 5）
 * @return 边缘上骨架点的连通簇数量，即该点的分支数
 *
 * 原理：在以 (cx,cy) 为中心、边长为 2*radius 的正方形边界上，
 * 找到所有骨架点，并将它们按 8 邻域连通聚成簇，簇的个数即为度数。
 * 这种方法比直接计算局部邻域更鲁棒，能避免骨架毛刺的干扰。
 */
int computeEdgeDegree(const Mat& skeleton, int cx, int cy, int radius = 5) {
    vector<Point> edgePoints;
    int rows = skeleton.rows, cols = skeleton.cols;

    // 采集方块四条边上的骨架点（避免重复）
    for (int dx = -radius; dx <= radius; ++dx) {
        // 上边
        int y = cy - radius, x = cx + dx;
        if (y >= 0 && y < rows && x >= 0 && x < cols && skeleton.at<uchar>(y, x) == 255)
            edgePoints.push_back(Point(x, y));
        // 下边
        y = cy + radius;
        if (y >= 0 && y < rows && x >= 0 && x < cols && skeleton.at<uchar>(y, x) == 255)
            edgePoints.push_back(Point(x, y));
    }
    for (int dy = -radius + 1; dy <= radius - 1; ++dy) {
        // 左边
        int x = cx - radius, y = cy + dy;
        if (y >= 0 && y < rows && x >= 0 && x < cols && skeleton.at<uchar>(y, x) == 255)
            edgePoints.push_back(Point(x, y));
        // 右边
        x = cx + radius;
        if (y >= 0 && y < rows && x >= 0 && x < cols && skeleton.at<uchar>(y, x) == 255)
            edgePoints.push_back(Point(x, y));
    }

    if (edgePoints.empty()) return 0;

    // 对边缘点进行 8 邻域聚类
    vector<vector<Point>> clusters;
    vector<bool> visited(edgePoints.size(), false);
    for (int i = 0; i < edgePoints.size(); ++i) {
        if (visited[i]) continue;
        // BFS 寻找连通簇
        vector<Point> cluster;
        queue<int> q;
        q.push(i);
        visited[i] = true;
        while (!q.empty()) {
            int cur = q.front(); q.pop();
            cluster.push_back(edgePoints[cur]);
            for (int j = 0; j < edgePoints.size(); ++j) {
                if (visited[j]) continue;
                int dx = abs(edgePoints[cur].x - edgePoints[j].x);
                int dy = abs(edgePoints[cur].y - edgePoints[j].y);
                if (dx <= 1 && dy <= 1) {
                    visited[j] = true;
                    q.push(j);
                }
            }
        }
        clusters.push_back(cluster);
    }
    return clusters.size();  // 簇数量即为度数
}

/**
 * @brief 构建骨架图（改进版，使用边缘度数识别分支点）
 * @param skeleton 输入骨架图像（二值，前景 255）
 * @param graph 输出的图结构
 * @param radius 计算边缘度数时的方块半径（默认 3）
 */
void buildSkeletonGraphV2(const Mat& skeleton, SkeletonGraph& graph, int radius = 3) {
    graph.pointToId.clear();
    graph.points.clear();
    graph.adj.clear();
    graph.degree.clear();
    graph.isJunction.clear();
    graph.isEndpoint.clear();

    // 收集所有骨架点
    vector<Point> allPoints;
    for (int y = 0; y < skeleton.rows; ++y)
        for (int x = 0; x < skeleton.cols; ++x)
            if (skeleton.at<uchar>(y, x) == 255)
                allPoints.push_back(Point(x, y));

    // 分配 ID
    for (auto& p : allPoints)
        graph.pointToId[p] = graph.points.size();
    graph.points = allPoints;
    int N = graph.points.size();
    graph.degree.assign(N, 0);
    graph.isJunction.assign(N, false);
    graph.isEndpoint.assign(N, false);
    graph.adj.resize(N);

    // 计算每个点的边缘度数
    for (int i = 0; i < N; ++i) {
        Point p = graph.points[i];
        int edgeDegree = computeEdgeDegree(skeleton, p.x, p.y, radius);
        graph.degree[i] = edgeDegree;
        if (edgeDegree == 1) graph.isEndpoint[i] = true;
        if (edgeDegree >= 3) graph.isJunction[i] = true;
    }

    // 建立邻接关系：8 邻域内相连的点之间添加边
    for (int i = 0; i < N; ++i) {
        Point p = graph.points[i];
        for (int k = 0; k < 8; ++k) {
            int nx = p.x + dx8[k], ny = p.y + dy8[k];
            if (nx >= 0 && ny >= 0 && nx < skeleton.cols && ny < skeleton.rows &&
                skeleton.at<uchar>(ny, nx) == 255) {
                auto it = graph.pointToId.find(Point(nx, ny));
                if (it != graph.pointToId.end()) {
                    int j = it->second;
                    if (i < j) {
                        graph.adj[i].insert(j);
                        graph.adj[j].insert(i);
                    }
                }
            }
        }
    }
}

// ---------------------- 4. 基于骨架分割粘连区域 ----------------------
/**
 * @brief 使用骨架分析将粘连的二值区域分割成多个独立细胞
 * @param binaryROI 输入二值图像（单个连通域，白色前景）
 * @return 分割后的多个二值掩膜（每个掩膜对应一个细胞）
 *
 * 算法步骤：
 * 1. 细化得到骨架
 * 2. 构建骨架图，找出所有度数为 3 的分支点
 * 3. 清理距离过近的分支点（<5 像素）
 * 4. 依次尝试阻断每个分支点（用圆形涂黑），若阻断后不会产生面积过小的碎片，则永久阻断
 * 5. 对最终骨架进行连通域分析，每个连通域对应一个细胞掩膜
 */
vector<Mat> splitBySkeleton(const Mat& binaryROI) {
    Mat skeleton;
    thinningZhangSuen(binaryROI, skeleton);   // 步骤1：细化

    // 构建骨架图
    SkeletonGraph graph;
    buildSkeletonGraphV2(skeleton, graph);
    cout << "骨架点数: " << graph.points.size() << endl;

    // 步骤2：收集所有度数为3的分支点
    vector<int> allJunctions;
    for (int i = 0; i < graph.points.size(); ++i) {
        if (graph.degree[i] == 3) {
            allJunctions.push_back(i);
        }
    }
    cout << "初始分支点数: " << allJunctions.size() << endl;

    // 步骤3：清理距离过近的分支点（5像素内只保留一个）
    vector<bool> removed(graph.points.size(), false);
    for (int i = 0; i < allJunctions.size(); ++i) {
        if (removed[allJunctions[i]]) continue;
        Point pi = graph.points[allJunctions[i]];
        for (int j = i + 1; j < allJunctions.size(); ++j) {
            if (removed[allJunctions[j]]) continue;
            Point pj = graph.points[allJunctions[j]];
            double dist = sqrt((pi.x - pj.x) * (pi.x - pj.x) + (pi.y - pj.y) * (pi.y - pj.y));
            if (dist < 5.0) {
                removed[allJunctions[j]] = true;
                cout << "移除邻近分支点: " << allJunctions[j] << " (距离 " << dist << ")" << endl;
            }
        }
    }

    vector<int> validJunctions;
    for (int j : allJunctions) {
        if (!removed[j]) validJunctions.push_back(j);
    }
    cout << "清理后分支点数: " << validJunctions.size() << endl;

    // 可视化分支点（调试用）
    Mat debugVis;
    cvtColor(skeleton, debugVis, COLOR_GRAY2BGR);
    for (int j : validJunctions) {
        Point p = graph.points[j];
        circle(debugVis, p, 6, Scalar(0, 0, 255), 2);
        putText(debugVis, to_string(j), p + Point(5, -5),
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 255), 1);
    }
    imshow("Branch Points - Red: candidate to split", debugVis);

    // 步骤4：尝试阻断分支点，避免产生小碎片
    Mat currentSkeleton = skeleton.clone();
    vector<int> blockedPoints;
    int blockRadius = 2;  // 阻断半径（像素）
    for (int pIdx : validJunctions) {
        Mat tempSkel = currentSkeleton.clone();
        Point p = graph.points[pIdx];
        circle(tempSkel, p, blockRadius, Scalar(0), -1);  // 涂黑该点及周围

        // 检查阻断后是否有面积过小的连通域（碎片）
        Mat labels;
        int numLabels = connectedComponents(tempSkel, labels, 8);
        bool hasSmall = false;
        const int MIN_AREA = 150;  // 最小碎片面积阈值
        for (int l = 1; l < numLabels; ++l) {
            Mat mask = (labels == l);
            if (countNonZero(mask) < MIN_AREA) {
                hasSmall = true;
                break;
            }
        }
        if (!hasSmall) {
            currentSkeleton = tempSkel.clone();
            blockedPoints.push_back(pIdx);
        }
        // 否则跳过，不阻断该分支点
    }

    // 步骤5：对最终骨架进行连通域分析，得到每个细胞的掩膜
    vector<Mat> resultMasks;
    Mat finalLabels;
    int numComponents = connectedComponents(currentSkeleton, finalLabels, 8);
    for (int l = 1; l < numComponents; ++l) {
        Mat mask = (finalLabels == l);
        // 可选：形态学闭运算填补缺口，然后与原区域取交集（此处未启用）
        if (countNonZero(mask) > 0) {
            resultMasks.push_back(mask);
        }
    }

    if (resultMasks.empty())
        resultMasks.push_back(binaryROI.clone());  // 未分割时返回原区域

    return resultMasks;
}

// ---------------------- 5. 主函数 ----------------------
/**
 * @brief 主函数：读取图像，对每个连通域进行骨架分割，输出可视化结果
 * @param argc 命令行参数个数
 * @param argv 命令行参数（可指定图像路径）
 * @return 0 表示成功
 */
int main(int argc, char** argv) {
    string imgPath = "object/cell.png";   // 默认图像路径，请根据实际情况修改
    if (argc > 1) imgPath = argv[1];

    Mat src = imread(imgPath, IMREAD_UNCHANGED);
    if (src.empty()) {
        cerr << "无法读取图像: " << imgPath << endl;
        return -1;
    }

    // 二值化（细胞白色，背景黑色）
    Mat bin = binarize(src);
    if (bin.type() != CV_8UC1) {
        bin.convertTo(bin, CV_8UC1);
    }

    // 获取所有连通域（每个粘连团单独处理）
    vector<vector<Point>> contours;
    findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat skeletonVis = Mat::zeros(bin.size(), CV_8UC3);   // 骨架可视化（绿色）
    Mat segmentationVis = Mat::zeros(bin.size(), CV_8UC3); // 分割结果（随机颜色）
    RNG rng(12345);
    int totalCells = 0;

    for (size_t i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        if (area < 500) continue;   // 忽略小噪点

        // 提取该连通域的 ROI（加边距）
        Rect roi = boundingRect(contours[i]);
        roi.x = max(roi.x - 5, 0);
        roi.y = max(roi.y - 5, 0);
        roi.width = min(roi.width + 10, bin.cols - roi.x);
        roi.height = min(roi.height + 10, bin.rows - roi.y);
        Mat roiBin = Mat::zeros(roi.size(), CV_8UC1);
        vector<vector<Point>> contourROI;
        vector<Point> shifted = contours[i];
        for (auto& pt : shifted) { pt.x -= roi.x; pt.y -= roi.y; }
        contourROI.push_back(shifted);
        drawContours(roiBin, contourROI, 0, Scalar(255), FILLED);

        // 骨架分割
        vector<Mat> cells = splitBySkeleton(roiBin);

        // 可视化该区域的骨架（绿色）
        Mat skelROI;
        thinningZhangSuen(roiBin, skelROI);
        // 可选：形态学闭运算平滑（此处未实际使用）
        for (int y = 0; y < skelROI.rows; ++y) {
            for (int x = 0; x < skelROI.cols; ++x) {
                if (skelROI.at<uchar>(y, x) == 255) {
                    skeletonVis.at<Vec3b>(roi.y + y, roi.x + x) = Vec3b(0, 255, 0);
                }
            }
        }

        // 分割结果着色（每个细胞随机颜色）
        for (const Mat& cellMask : cells) {
            vector<vector<Point>> cnts;
            findContours(cellMask, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            if (cnts.empty()) continue;
            Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            for (const auto& c : cnts) {
                vector<Point> c_global = c;
                for (auto& pt : c_global) { pt.x += roi.x; pt.y += roi.y; }
                drawContours(segmentationVis, vector<vector<Point>>{c_global}, 0, color, FILLED);
            }
            totalCells++;
        }
    }

    // 显示并保存结果
    imshow("Original", src);
    imshow("Skeleton (green)", skeletonVis);
    imshow("Segmentation (color)", segmentationVis);
    imwrite("output/skeleton.jpg", skeletonVis);
    imwrite("output/segmentation.jpg", segmentationVis);
    cout << "分割完成，共得到 " << totalCells << " 个细胞" << endl;

    waitKey(0);
    destroyAllWindows();
    return 0;
}