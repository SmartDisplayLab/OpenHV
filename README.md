# OpenHV Software Development Documentation – HumanVisionSimulatorUI

This document primarily introduces the software used in Paper [Bionic Vision Processing for Epiretinal Implant-Based Metaverse](https://pubs.acs.org/doi/full/10.1021/acsaom.3c00431) and related papers.

🌐 Available Languages: [English](README.md) | [简体中文](README.zh-CN.md)

## Development Tools Installation

### Python IDE Installation

1. PyCharm Community Edition 2024.1.3: Install from the official website.

2. Anaconda virtual environment: Download and install from the official website.

3. Use Anaconda virtual environment in PyCharm: Remember the Anaconda installation path, and set it in the Interpreter settings at the bottom-right corner of PyCharm.

### Unity Installation

[Download Unity from the official website | Unity Hub Installation | Unity China](https://unity.cn/releases)

Unity version: 2021.3.8f1c1

## Software Instructions

This section describes how to use the software and the functions of each module.

### Installation

```
git clone
cd HumanVisionSimulatorUI
conda env create -f environment.yml
conda activate HumanVision
```

### Run

```
pyhton main.py  --base_dir path_to_project
                --unity_path path_to_Unity
                --project_path path_to_Unity_project
```


### Functional Modules Description

#### Start

1. Input parameters: binocular focus length, position, type of focus, FOV, and pupil length.

2. Input images: Import left and right eye images at "Left Eye Image Location" and "Right Eye Image Location", or generate them in Unity. The input images will be displayed below.

#### Blur and Mask

1. 2D Part: Apply blur to the images and mask them with binocular vision limitations.
2. 3D Part: Show the projection of the images on the retina (back of the eyeball), with an adjustable axial radius ratio to change axial length.

#### Binocular Fusion

Monocular or binocular image fusion.

#### Depth Map

Depth map visualization.

#### Edge Detection

Edge detection module.

#### Saliency Detection

Saliency detection module.

## Code Structure

This section introduces the code structure and the functions of each component.

### File Descriptions

- `main.py`: Entry point of the software. Running this file launches the software. The `__init__` method of the `MyWindow` class contains the main logic. Other methods implement specific functionalities.

- `HV.py`: Main functions for image processing algorithms.

- `ImageProcessFunction.py`: Implementation of retinal blurring and image fusion algorithms.

- `CorrectionFunction.py`: Implementation of epipolar rectification algorithms.

- `DepthDetection.py`: Implementation of the SGBM algorithm.

- `V1_Function.py`: Implementation of the edge detection algorithm.

- `xianzhuxing.py`: Implementation of the saliency detection algorithm.
