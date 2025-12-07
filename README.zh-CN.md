# OpenHV软件开发文档——HumanVisionSimulatorUI

Open Human Vision (OpenHV) 是一个开源工具库，用于模拟与分析人类视觉系统在 2D 与 3D 场景中的工作方式。
它集成了双眼图像处理、视网膜投影、模糊模拟、深度估计、图像融合与显著性分析等多种模块，方便研究人员、工程师和开发者快速搭建、可视化与测试与人类视觉相关的算法与显示技术。

OpenHV 致力于提供一个模块化、可扩展的框架，支持人类视觉建模、计算神经科学、VR/AR 显示技术评估以及仿生视觉系统等方向的研究。

🌐 可用语言: [English](README.md) | [简体中文](README.zh-CN.md)

## 开发工具安装

### Python IDE安装

1. PyCharm Community Edition 2024.1.3：官网安装

2. anaconda虚拟环境安装：官网下载安装

3. 在pycharm中使用anaconda虚拟环境：记住anaconda的路径，并在pycharm右下角的Interpreter setting中设置

### Unity安装

[Unity官方下载_Unity新版_从Unity Hub下载安装 | Unity中国官网](https://unity.cn/releases)

Unity版本为2021.3.8f1c1

## 软件说明

本部分介绍软件的使用方法和各部分功能。

### 安装

```
git clone
cd OpenHV
conda env create -f environment.yml
conda activate HumanVision
```

### 运行

```
pyhton main.py  ----config 配置文件
```

### 软件各部分功能说明

#### Start

   1. 输入参数：binocular focus length，position，type of focus，FOV和pupil length. 

   2. 输入图像：在Left Eye Image Location和Right Eye Image Location处输入左右眼图片（可使用example_photos中示例图片）或在Unity中生成。输入的图像会显示在下方.
      
#### Bulr and Mask

   1. 2D部分：对图像进行模糊，并施加双眼视觉限制的掩膜。
   2. 3D部分：显示图像在眼球后半部的视网膜上的投影,可调整axial radius ratio以改变眼轴长度。

#### Binocular Fusion

    单边、双边双目融合

#### Depth Map

    深度图

#### Edge Detection

    边缘检测

#### Saliency Detection

    显著性检测

## 代码说明

本部分介绍软件的代码结构和各部分功能。

### 各文件功能

main.py: 运行即可出现软件，MyWindow类的__init__方法包括了主要运行逻辑，后续方法实现了各部分功能逻辑。

HV.py: 软件各图像算法功能主函数

ImagePrcessFunction.py: 视网膜化模糊和图像融合算法实现

CorrectionFunction.py：极线校正算法实现

DepthDetection.py：SGBM算法

V1_Function.py：边缘检测算法实现

saliency.py：显著性算法实现
