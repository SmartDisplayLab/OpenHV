import cv2
import numpy as np
import matplotlib.pyplot as plt
from CorrectionFunction import epipolarCorrection
from math import tan, pi

ring1 = cv2.imread('fig/mask1.png')
ring2 = cv2.imread('fig/mask2.png')
ring3 = cv2.imread('fig/mask3.png')
ring4 = cv2.imread('fig/mask4.png')


# 对img加入形如mask的遮罩
def add_mask(img, mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.add(img, np.zeros(np.shape(img), dtype=img.dtype), mask=mask)
    return img


# 旧模糊函数
# def blured_img(img):
#     kernel_sizes = [1, 3, 7, 21]
#     img_copy = img.copy()
#
#     tmp1 = cv2.GaussianBlur(img_copy, ksize=(1, 1), sigmaX=0)
#     mask1 = 255 - cv2.cvtColor(ring1, cv2.COLOR_BGR2GRAY)
#
#     tmp2 = cv2.GaussianBlur(img_copy, ksize=(3, 3), sigmaX=0)
#     mask2 = 255 - cv2.cvtColor(ring2, cv2.COLOR_BGR2GRAY)
#
#     tmp3 = cv2.GaussianBlur(img_copy, ksize=(7, 7), sigmaX=0)
#     mask3 = 255 - cv2.cvtColor(ring3, cv2.COLOR_BGR2GRAY)
#
#     tmp4 = cv2.GaussianBlur(img_copy, ksize=(33, 33), sigmaX=0)
#     mask4 = 255 - cv2.cvtColor(ring4, cv2.COLOR_BGR2GRAY)
#
#     img1 = cv2.add(tmp1, img_copy, mask=mask1)
#     img2 = cv2.add(tmp2, img_copy, mask=mask2)
#     img3 = cv2.add(tmp3, img_copy, mask=mask3)
#     img4 = cv2.add(tmp4, img_copy, mask=mask4)
#
#     img = img1 + img2 + img3 + img4
#
#     return img


# 差异化模糊
def blured_img(img):
    blur = [47, 53, 60, 67, 75, 86, 97, 111,
            127, 148, 180, 214, 274, 401, 631]
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    num = 14
    center = (width // 2, height // 2)
    img_result = np.zeros_like(img)
    for i in range(num):
        inner_radius = int(blur[i] * width / 631)
        outer_radius = int(blur[i + 1] * width / 631)
        cv2.circle(mask, center, outer_radius if i < 14 else width, (255, 255, 255), -1)
        cv2.circle(mask, center, inner_radius if i > 0 else 0, (0, 0, 0), -1)
        n = i * 2 + 1
        blurred_img = cv2.GaussianBlur(img, (n, n), 10)
        img_result = cv2.bitwise_and(blurred_img, mask) + img_result
    return img_result


# 对输入mask加入盲点，参数可调
def add_blind(mask, side='left'):
    eye_diameter = 48  # 这个需要调整
    h_fov = 150
    real_w = 2 * eye_diameter * tan(pi * h_fov / 360)  # 公式来源待定？

    W = mask.shape[1]
    H = mask.shape[0]
    # 确定盲点位置
    blind_hole_dis = 3
    blind_hole_r = 1.5
    b_w_dis = tan(blind_hole_dis / eye_diameter) * eye_diameter * W / real_w  # ？
    if side == 'right':
        center_blind = [int(H / 2), int(W / 2 - b_w_dis)]
    else:
        center_blind = [int(H / 2), int(W / 2 + b_w_dis)]
    r_blind = int(0.5 * blind_hole_r * W / real_w)

    # 在掩膜上抠去盲点部分
    cv2.circle(mask, (center_blind[1], center_blind[0]), r_blind, (0, 0, 0), -1)
    # print(center_blind)
    # print(r_blind)
    # 当前参数下的结果
    # [930, 961]
    # 7
    # [930, 898]
    # 7
    return mask


def cat_two_images(l, r):
    return np.concatenate([l, r], axis=1)


def unlateral_fusion_SIFT(a, b, side='left'):
    hessian = 400
    surf = cv2.SIFT_create(hessian)  # 将Hessian Threshold设置为400,阈值越大能检测的特征就越少
    kp1, des1 = surf.detectAndCompute(a, None)  # 查找关键点和描述符
    kp2, des2 = surf.detectAndCompute(b, None)

    FLANN_INDEX_KDTREE = 0  # 建立FLANN匹配器的参数
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 配置索引，密度树的数量为5
    searchParams = dict(checks=50)  # 指定递归次数
    # FlannBasedMatcher：是目前最快的特征匹配算法（最近邻搜索）
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)  # 建立匹配器
    matches = flann.knnMatch(des1, des2, k=2)  # 得出匹配的关键点

    good = []
    # 提取优秀的特征点
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # 如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
            good.append(m)

    src_pts = np.array([kp1[m.queryIdx].pt for m in good])  # 查询图像的特征描述子索引
    dst_pts = np.array([kp2[m.trainIdx].pt for m in good])  # 训练(模板)图像的特征描述子索引
    H = cv2.findHomography(src_pts, dst_pts)  # 生成变换矩阵

    h, w = a.shape[:2]
    h1, w1 = b.shape[:2]
    shft = np.array([[1.0, 0, w], [0, 1.0, 0], [0, 0, 1.0]])
    M = np.dot(shft, H[0])  # 获取左边图像到右边图像的投影映射关系
    print(M)

    if side == 'right':
        # M = np.array([[-1.18520638e+00, -6.48066337e-01, 1.72492679e+03],
        #               [-4.97797789e-01, -2.74502861e-01, 7.26482605e+02],
        #               [-6.87201605e-04, -3.75947364e-04, 1.00000000e+00]])
        dst_corners = cv2.warpPerspective(a, M, (w * 2, h))  # 透视变换，新图像可容纳完整的两幅图
        dst_corners[0:h, w:w * 2] = b  # 将第二幅图放在右侧
    else:
        # M = np.array([[1.01258743e+01, -2.14302211e+00, 9.39487968e+02],
        #               [8.57189730e+00, -1.69251301e+00, 6.64840857e+02],
        #               [9.50919680e-03, -2.12305572e-03, 1.00000000e+00]])
        dst_corners = cv2.warpPerspective(b, np.linalg.inv(M), (w * 2, h))  # 透视变换，新图像可容纳完整的两幅图
        # dst_corners[0:h, :w] = a  # 将第一幅图放在左侧
    return dst_corners


# def unlateral_fusion(a, b, A, B, side='left'):
#     h, w = A.shape
#     if side == 'right':
#         fusion = None
#     else:
#         fusion = None
#     return fusion


def image_fusion(L, R, shift):
    """
    基于水平偏移的左右图像融合
    Args:
        L, R: 左右输入图像 (numpy arrays, same size)
        shift: 水平偏移像素（正值表示右图向右移动）
    Returns:
        r_cat: 最终融合图像
        r_cat_L: 左半部分融合
        r_cat_R: 右半部分融合
    """
    h, w = L.shape[:2]
    shift = np.clip(shift, -w//2, w//2)  # 防止越界

    # 分块
    ll, lr = L[:, :w//2], L[:, w//2:]
    rl, rr = R[:, :w//2], R[:, w//2:]

    # 创建空图
    zeros = np.zeros_like(L)

    # 计算位置索引
    def place_region(base, region, x_start):
        x_end = x_start + region.shape[1]
        if x_end <= 0 or x_start >= w:  # 完全超出范围
            return base
        xs, xe = max(0, x_start), min(w, x_end)
        rs, re = max(0, -x_start), region.shape[1] - max(0, x_end - w)
        base[:, xs:xe] = region[:, rs:re]
        return base

    # 平移放置
    LL = place_region(zeros.copy(), ll, 0)
    LR = place_region(zeros.copy(), lr, w//2 - shift)
    RL = place_region(zeros.copy(), rl, shift)
    RR = place_region(zeros.copy(), rr, w//2)

    # 融合
    r_cat_L = cv2.addWeighted(RL, 0.5, LL, 0.5, 0)
    r_cat_R = cv2.addWeighted(RR, 0.5, LR, 0.5, 0)

    # 合并
    validR = np.where(np.sum(r_cat_R, axis=2) > 0)
    x_min = np.min(validR[1]) if validR[0].size > 0 else w//2

    r_cat = np.zeros_like(L)
    r_cat[:, :x_min] = r_cat_L[:, :x_min]
    r_cat[:, x_min:] = r_cat_R[:, x_min:]

    return r_cat, r_cat_L, r_cat_R


# cv2.imwrite('tmp.png', blured_img(img_l))
# mask_L = cv2.imread('fig/mask-82-l.png')
# maskL = cv2.resize(mask_L, (1860, 1860))
# maskR = cv2.flip(maskL, 1)
#
# L, R = add_mask(img_l, img_r, maskL, maskR)
# cv2.imwrite('L.png', L)
# cv2.imwrite('R.png', R)


if __name__ == '__main__':
    from setting import *

    mask_r = cv2.imread(f"fig/NEWmask_r_164.png")
    mask_l = np.flip(mask_r, axis=1)
    h, w, _ = mask_r.shape

    fov = 150
    pupil_length = 0.06
    FocusLength = 0.4
    pos = 0

    new_h = int(np.tan(fov / 360 * np.pi) / np.tan(164 / 360 * np.pi) * h)
    new_w = int(np.tan(fov / 360 * np.pi) / np.tan(164 / 360 * np.pi) * w)
    mask_r = mask_r[h // 2 - new_h // 2:h // 2 + new_h // 2, w // 2 - new_w // 2:w // 2 + new_w // 2]
    mask_l = mask_l[h // 2 - new_h // 2:h // 2 + new_h // 2, w // 2 - new_w // 2:w // 2 + new_w // 2]

    maskL_noBlind = cv2.resize(mask_l, (1860, 1860))
    maskR_noBlind = cv2.resize(mask_r, (1860, 1860))
    maskL = add_blind(copy.deepcopy(maskL_noBlind), side='left')
    maskR = add_blind(copy.deepcopy(maskR_noBlind), side='right')

    img_l = cv2.imread(f"fig_tmp/left_FOV{fov}-F{FocusLength}-pos{pos}.jpg")
    img_r = cv2.imread(f"fig_tmp/right_FOV{fov}-F{FocusLength}-pos{pos}.jpg")

    # cv2.imwrite('tmp_l_ori.png', img_l)
    img_l = blured_img(img_l)
    # cv2.imwrite('tmp_l.png', img_l)

    img_r = blured_img(img_r)
    img_l = add_mask(img_l, maskL)
    img_r = add_mask(img_r, maskR)

    h_fov = v_fov = fov
    hh = math.atan(pupil_length / 2 / FocusLength) * 180 / math.pi
    right_euler_angle = [0, hh, 0]
    left_euler_angle = [0, -hh, 0]

    # Do correction
    resL, resR = epipolarCorrection(img_l, img_r, right_euler_angle,
                                    left_euler_angle, camera_focus_length, pupil_length,
                                    h_fov, v_fov)

    # r = image_fusion(img_l, img_r)
    # shift = 12
    shift = int(12 * (math.tan(fov/2/180*math.pi) / math.tan(80/180*math.pi)))

    h, w, _ = img_l.shape
    ll = resL[:, :w // 2, :]
    LL = np.zeros_like(img_l)
    LL[:, :w // 2, :] = ll
    lr = resL[:, w // 2:, :]
    LR = np.zeros_like(img_l)
    LR[:, -shift + w // 2:-shift + w, :] = lr
    rl = resR[:, :w // 2, :]
    RL = np.zeros_like(img_l)
    RL[:, shift:shift + w // 2, :] = rl
    rr = resR[:, w // 2:, :]
    RR = np.zeros_like(img_l)
    RR[:, w // 2:, :] = rr

    # l_cat = unlateral_fusion(ll, rl, side='left')
    # r_cat = unlateral_fusion(lr, rr, side='right')
    # r = image_fusion(resL, resR)

    r_cat = np.zeros_like(img_l)

    lr_gray = cv2.cvtColor(lr, cv2.COLOR_BGR2GRAY)
    rr_gray = cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY)
    r_cat_gray = cv2.cvtColor(r_cat, cv2.COLOR_BGR2GRAY)

    ind_mask_lr = np.where(lr_gray > 0)
    ind_mask_rr = np.where(rr_gray > 0)
    mask_lr = np.zeros_like(lr_gray)
    mask_rr = np.zeros_like(rr_gray)
    mask_lr[ind_mask_lr] = 255
    mask_rr[ind_mask_rr] = 255
    mask_lr_total = np.zeros_like(r_cat_gray)
    mask_lr_total[:, -shift + w // 2:-shift + w] = mask_lr
    mask_rr_total = np.zeros_like(r_cat_gray)
    mask_rr_total[:, w // 2:w] = mask_rr

    mask_lr_total = mask_lr_total - mask_rr_total

    mask_ll_total = np.flip(mask_rr_total, axis=1)

    mask_rl_total = np.flip(mask_lr_total, axis=1)

    # print(LR.shape, mask_lr_total.shape)
    r_cat_LR = cv2.add(np.zeros_like(LR), LR,
                     mask=mask_lr_total)
    r_cat_RR = cv2.add(np.zeros_like(RR), RR,
                     mask=mask_rr_total)
    r_cat_RL = cv2.add(np.zeros_like(RL), RL,
                     mask=mask_rl_total)
    r_cat_LL = cv2.add(np.zeros_like(LL), LL,
                     mask=mask_ll_total)
    r_cat_L = r_cat_RL + r_cat_LL
    r_cat_R = r_cat_RR + r_cat_LR
    validR = np.where(r_cat_R > 0)
    x_min = np.min(validR[1])

    r_cat[:, x_min:, :] = r_cat_R[:, x_min:, :]
    r_cat[:, :x_min, :] = r_cat_L[:, :x_min, :]
    cv2.imwrite('tmp_l.png', r_cat)
    cv2.imwrite('tmp_r.png', r_cat_L)
    # cv2.arrowedLine(resL, (w//2, 0), (w//2, h), color=(0, 0, 0))
    # cv2.arrowedLine(resR, (w // 2, 0), (w // 2, h), color=(0, 0, 0))

    # cv2.imwrite('tmp_l_ori.png', ll)
    # cv2.imwrite('tmp_r_ori.png', lr)
    # cv2.imwrite('tmp_l.png', r_cat_L)
    # cv2.imwrite('tmp_r.png', r_cat_R)
    #
    # cv2.imwrite('tmp.png', r_cat)
