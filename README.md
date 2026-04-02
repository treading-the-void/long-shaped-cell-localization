# 长形细胞定位与分割工具

基于 OpenCV 的显微图像处理工具，用于长形细胞的检测、分割与定位。支持实时参数调节、基于骨架分析的粘连细胞分离、Harris 角点检测（自动识别细胞头尾）以及视频批处理。

## ✨ 功能特性

| 模块 | 功能 |
|------|------|
| **实时参数调优** | 通过滑动条动态调整中值滤波、高斯滤波、Canny阈值、形态学核大小，即时查看效果 |
| **单细胞分割** | 基于轮廓检测和面积筛选，自动提取单个细胞区域并保存 |
| **骨架粘连分离** | Zhang-Suen细化 + 骨架图分析，在分支点处智能切断，分离粘连细胞 |
| **角点检测** | Harris角点检测，自动标记细胞最远两端（头部/尾部）并输出坐标 |
| **视频处理** | 支持跳帧采样、边缘检测、细胞标记，输出处理后的视频文件 |

## 🖥 系统要求

- **操作系统**：Windows 10 / 11
- **编译器**：Visual Studio 2022（或支持 C++17 的编译器）
- **依赖库**：OpenCV 4.x（已配置好包含目录和库目录）

## 📦 模块说明

| 源文件 | 功能描述 |
|--------|----------|
| `parameter_adjustment.cpp` | 交互式参数调节工具，滑动条实时控制滤波、Canny、形态学参数 |
| `picture.cpp` | 单张图像处理：预处理 → 轮廓分割 → Harris角点检测 → 标记头尾 |
| `skeleton_separation.cpp` | 基于骨架分析的粘连细胞分离算法（Zhang-Suen细化 + 分支点切割） |
| `video.cpp` | 视频处理：跳帧采样、边缘检测、轮廓标记、输出结果视频 |
| `imgFeat.h` | Harris角点检测与绘制的函数声明 |

## 📖 使用指南

### 文件一：参数调优（`parameter_adjustment.cpp`）

运行后出现两个窗口：
- **Controls**：包含 7 个滑动条，实时调整参数
- **Image**：显示处理后的二值边缘图像

**操作流程**：
1. 修改 `object/110RAY2.jpg` 为你的图片路径
2. 拖动滑动条，观察图像变化
3. 找到最佳参数组合后，按 `ESC` 退出

### 文件二：单细胞分割与角点检测（`picture.cpp`）

运行后自动执行：
1. 预处理（中值滤波 → 高斯滤波 → Canny → 形态学闭/开运算）
2. 轮廓检测，筛选面积 ≥ 1500 的区域
3. 对每个细胞区域：
   - Harris 角点检测（alpha = 0.12）
   - 找出距离最远的两个角点（头部/尾部）
   - 在图像上绘制并保存到 `output/` 文件夹

**输出文件**：
- `output/singlecell0.jpg`, `singlecell1.jpg`, ... 每个细胞的标注图像

### 文件三：骨架粘连分割（`skeleton_separation.cpp`）

适用于粘连严重的细胞图像。算法步骤：
1. 二值化
2. 对每个连通域进行 Zhang-Suen 细化得到骨架
3. 构建骨架图，识别度数 ≥ 3 的分支点
4. 清理距离过近的分支点（< 5 像素）
5. 在分支点处"阻断"骨架，避免产生小碎片
6. 连通域分析，每个连通域对应一个细胞

**输出**：
- `output/skeleton.jpg`：骨架可视化（绿色）
- `output/segmentation.jpg`：分割结果（随机颜色）

### 模式四：视频处理（`video.cpp`）

处理 `object/Ray 1.mp4` 视频：
- 高斯滤波 → Canny → 膨胀腐蚀 → 轮廓检测 → 绘制矩形框
- 输出文件：`output/video_output.mp4`

## 📁 文件结构

```
long-shaped_cell_localization/
│
├── .gitignore                          # Git 忽略规则
├── long-shaped_cell_localization.sln   # VS 解决方案文件
├── README.md                           # 项目说明
│
├── parameter_adjustment.cpp            # 实时参数调优
├── picture.cpp                         # 单细胞分割+角点检测
├── skeleton_separation.cpp             # 骨架粘连分离
├── video.cpp                           # 视频处理
├── imgFeat.h                           # Harris 角点检测头文件
│
├── object/                             # 测试数据目录
│   ├── 110RAY2.jpg                     # 测试图片1
│   ├── 366RAY6.jpg                     # 测试图片2
│   ├── 39RAY5.jpg                      # 测试图片3
│   ├── cell.png                        # 测试图片4
│   └── Ray 1.mp4                       # 测试视频
│
└── output/                             # 程序输出目录（自动创建）
    ├── singlecell0.jpg                 # 单细胞标注结果
    ├── skeleton.jpg                    # 骨架可视化
    ├── segmentation.jpg                # 分割结果
    └── video_output.mp4                # 处理后视频
```

## 📄 额外说明与感悟

- 大二起始做这个项目，我对做项目还没有任何概念，AI尚未盛行，啃OpenCV就用了好久，代码一行行打，顶多参考下CSDN。我不知道学新东西要多长时间，所以python和机器学习都没敢碰。
- 传统视觉识别确实有很多限制，花了很多时间调参，用处不大，但凡换个尺寸的图片，原来的算子就不适用了。视频处理的速度也慢，效果也不佳。
- 难以想象这个项目断断续续做了一年，在时代和自己都成长过后，恐怕两三天就能做完。
- 当时没解决110RAY2.jpg中两个细胞粘连的问题，现在借助AI的力量试了一天，有效果，但用了点取巧（糊弄）的方法。AI你继续发展吧。


## 👤 作者

- GitHub: [@treading-the-void](https://github.com/treading-the-void)

