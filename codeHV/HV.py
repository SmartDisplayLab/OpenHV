import math

from DepthDetectionFunction import depthMap
from V1_Function import *
from saliency import Visual_Saliency_Detection
from setting import *


def f0(l, r, lf, rf):
    img_l = cv2.imread(l)
    img_r = cv2.imread(r)
    cv2.imwrite(lf, img_l)
    cv2.imwrite(rf, img_r)
    return img_l, img_r


# 模糊，视野FOV限制（包含盲点）
def blur(l, r, lf, rf, maskL, maskR):
    img_l = cv2.imread(l, cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(r, cv2.IMREAD_GRAYSCALE)

    img_l = blured_img(img_l)
    img_r = blured_img(img_r)
    img_l = add_mask(img_l, maskL)
    img_r = add_mask(img_r, maskR)
    img_l = cv2.resize(img_l, show_size)
    img_r = cv2.resize(img_r, show_size)
    cv2.imwrite(lf, img_l)
    cv2.imwrite(rf, img_r)


# 单边、双边双目融合
def binocular_fusion(l, r, f, f1, f2, fov, pupil_length, FocusLength, maskL, maskR):
    h_fov = v_fov = fov
    hh = math.atan(pupil_length/2/FocusLength) * 180 / math.pi
    right_euler_angle = [0, hh, 0]
    left_euler_angle = [0, -hh, 0]

    #img_l = cv2.imread(l, cv2.IMREAD_GRAYSCALE)
    if isinstance(l, str):
        img_l = cv2.imread(l, cv2.IMREAD_GRAYSCALE)
    elif isinstance(l, np.ndarray):
        img_l = l
    else:
        raise TypeError("image_source must be path or bytes")
    #img_r = cv2.imread(r, cv2.IMREAD_GRAYSCALE)
    if isinstance(r, str):
        img_r = cv2.imread(r, cv2.IMREAD_GRAYSCALE)
    elif isinstance(r, np.ndarray):
        img_r = r
    else:
        raise TypeError("image_source must be path or bytes")

    img_l = blured_img(img_l)
    img_r = blured_img(img_r)
    img_l = add_mask(img_l, maskL)
    img_r = add_mask(img_r, maskR)

    # Do correction
    resL, resR = epipolarCorrection(img_l, img_r, right_euler_angle,
                                    left_euler_angle, camera_focus_length, pupil_length,
                                    h_fov, v_fov)
    
  
    r, l_cat, r_cat = image_fusion(resL, resR, shift=int(12 * (math.tan(fov/2/180*math.pi) / math.tan(80/180*math.pi))))
    l_cat = cv2.resize(l_cat, show_size)
    r_cat = cv2.resize(r_cat, show_size)
    r = cv2.resize(r, show_size)

    cv2.imwrite(f1, l_cat)
    cv2.imwrite(f2, r_cat)
    cv2.imwrite(f, r)

    return r


# 深度图结果
def compute_depth_map(l, r, f, fov, pupil_length, FocusLength, maskL, maskR):
    h_fov = v_fov = fov
    hh = math.atan(pupil_length / 2 / FocusLength) * 180 / math.pi
    right_euler_angle = [0, hh, 0]
    left_euler_angle = [0, -hh, 0]

    img_l = cv2.imread(l, cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(r, cv2.IMREAD_GRAYSCALE)

    img_l = blured_img(img_l)
    img_r = blured_img(img_r)
    img_l = add_mask(img_l, maskL)
    img_r = add_mask(img_r, maskR)

    resL, resR = epipolarCorrection(img_l, img_r, right_euler_angle,
                                    left_euler_angle, camera_focus_length, pupil_length,
                                    h_fov, v_fov)

    maskL, maskR = epipolarCorrection(maskL, maskR, right_euler_angle,
                                      left_euler_angle, camera_focus_length, pupil_length,
                                      h_fov, v_fov)

    if len(maskL.shape) == 2:
        maskL = cv2.cvtColor(maskL, cv2.COLOR_GRAY2BGR)
        maskR = cv2.cvtColor(maskR, cv2.COLOR_GRAY2BGR)

    mask = cv2.bitwise_and(maskL, maskR)

    # depthImg = depthMap(resL, resR, mask)
    depthImg = depthMap(resL, resR, mask)
    cv2.imwrite('./fig/depth_map.jpg',depthImg)
    depthImg = cv2.resize(depthImg, show_size)

    plt.figure()
    cax = plt.imshow(depthImg)
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar(cax, ticks=[0, 48, 96, 144, 192, 240])
    cbar.ax.set_yticklabels(['0 m', '3 m', '6 m', '9 m', '12 m', '15 m'])

    plt.savefig(f)


# 边缘检测
def edge_detection(input, f):
    img = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    img = unsharp_mask1(img)
    img = edge_detection_Canny(img)
    img = cv2.resize(img, show_size)
    cv2.imwrite(f, img)


# 显著性检测
def segment_saliency(input, f):
    img = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    tmp = Visual_Saliency_Detection(img)
    tmp = cv2.resize(tmp, show_size)
    cv2.imwrite(f, tmp)
