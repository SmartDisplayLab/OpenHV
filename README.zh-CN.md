# OpenHV软件开发文档——HumanVisionSimulatorUI

本文档主要介绍了论文[Bionic Vision Processing for Epiretinal Implant-Based Metaverse](https://pubs.acs.org/doi/full/10.1021/acsaom.3c00431)及其相关论文中使用的软件。

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
cd HumanVisionSimulatorUI
conda env create -f environment.yml
conda activate HumanVision
```

### 运行

```
pyhton main.py  --base_dir 项目路径
                --unity_path Unity路径
                --project_path Unity项目路径
```

### 软件各部分功能说明

#### Start

   1. 输入参数：binocular focus length，position，type of focus，FOV和pupil length. 

   2. 输入图像：在Left Eye Image Location和Right Eye Image Location处输入左右眼图片或在Unity中生成。输入的图像会显示在下方.
      
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

xianzhuxing.py：显著性算法实现
