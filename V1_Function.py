import cv2
import numpy as np
import matplotlib.pyplot as plt
from CorrectionFunction import epipolarCorrection
from codeHV.ImageProcessFunction import *
import copy


def unsharp_mask1(image, kernel_size=(3, 3), sigma=2, amount=3.0, threshold=2):
    # 使用高斯滤波器对图像进行模糊处理
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)

    # 计算模糊后的图像与原图像的差值
    sharpened = float(amount + 1) * image - float(amount) * blurred

    # 将差值限制在0-255范围内
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)

    # 应用阈值来创建掩膜
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    return sharpened


def edge_detection_Canny(img, lt=60, ht=120):
    dst = cv2.Canny(img, lt, ht)
    return dst


def edge_detection_Prewitt(img):
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(img, cv2.CV_16S, kernelx)
    y = cv2.filter2D(img, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst


def edge_detection_Log(img):
    # 高斯滤波
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    # 拉普拉斯算子
    laplacian = cv2.Laplacian(img_blur, cv2.CV_64F)
    # 对数变换
    dst = np.uint8(np.log(laplacian + 1) / (np.log(1 + np.max(laplacian)) / 255))
    return dst


if __name__ == '__main__':
    # img_l = cv2.imread("result0.png", cv2.IMREAD_GRAYSCALE)
    img_l = cv2.imread("result1.png")
    # img_l = cv2.resize(img_l, (960, 960))

    # mask_r = cv2.imread(f"fig/NEWmask_r_164.png")
    # mask_l = np.flip(mask_r, axis=1)
    # h, w, _ = mask_r.shape
    #
    # fov = 150
    # new_h = int(np.tan(fov / 360 * np.pi) / np.tan(164 / 360 * np.pi) * h)
    # new_w = int(np.tan(fov / 360 * np.pi) / np.tan(164 / 360 * np.pi) * w)
    # mask_r = mask_r[h // 2 - new_h // 2:h // 2 + new_h // 2, w // 2 - new_w // 2:w // 2 + new_w // 2]
    # mask_l = mask_l[h // 2 - new_h // 2:h // 2 + new_h // 2, w // 2 - new_w // 2:w // 2 + new_w // 2]
    #
    # maskL_noBlind = cv2.resize(mask_l, img_l.shape)
    # maskR_noBlind = cv2.resize(mask_r, img_l.shape)
    # maskL = add_blind(copy.deepcopy(maskL_noBlind), side='left')
    # maskR = add_blind(copy.deepcopy(maskR_noBlind), side='right')
    img_l = blured_img(img_l)
    # img_l = add_mask(img_l, maskL)

    # img_l = unsharp_mask1(img_l)
    # img_l = edge_detection_Canny(img_l)
    # img_l = cv2.resize(img_l, (360, 360))
    cv2.imwrite('tmp.png', img_l)
