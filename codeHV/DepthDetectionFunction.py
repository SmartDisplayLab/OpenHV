import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ImageProcessFunction import *
from setting import *


def depthMap(imgL, imgR, mask=None):
    # imgL = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
    # imgR = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)
    # disparity range tuning
    window_size = 3
    min_disp = 0
    # num_disp = 320 - min_disp
    if mask is not None:
        imgLM = add_mask(imgL, mask)
        imgRM = add_mask(imgR, mask)
    else:
        imgLM = imgL
        imgRM = imgR

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=240,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image;
        # 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity = stereo.compute(imgLM, imgRM).astype(np.float32)

    # a = np.reshape(disparity, -1)
    # a = np.sort(a)
    upperlimit = 150
    # lowerlimit = a[np.where(a > 0)]
    lowerlimit = 20

    # a = np.reshape(disparity, -1)
    # plt.hist(a, bins=[0, 0.1, 50, 100, 150, 200, 250, 300, 10000])
    # plt.semilogy()
    # plt.semilogx()
    # plt.title("histogram")
    # plt.show()
    #
    print(np.median(disparity), np.max(disparity), np.min(disparity))
    # plt.hist(np.reshape(disparity, -1), bins=30)
    # plt.show()
    disparity = np.clip(disparity, a_max=240, a_min=0)
    # disparity = np.clip(255 * (disparity - lowerlimit) / (upperlimit - lowerlimit), a_max=255, a_min=0)
    #
    # print(np.median(disparity), np.max(disparity), np.min(disparity))
    # plt.hist(np.reshape(disparity, -1), bins=30, log=True)
    # plt.show()

    mask_tmp = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    disparity[np.where(mask_tmp == 0)] = 0
    distance = (0.06 * 0.2491928) / (disparity * 0.186 / 1860 + 1e-6) * 4
    # print(np.max(distance[np.where(distance < 255)]))
    # # distance = 1 / (disparity + 1e-6)
    farClip = 15
    distance = np.clip(distance, a_max=farClip, a_min=0.2)
    # # # print(np.max(distance[np.where(distance < 255)]))
    # # # distance = np.log(distance)-
    # distance = 255 * (distance - np.min(distance)) / (np.max(distance) - np.min(distance))
    distance = 255 / farClip * distance
    # print(np.median(distance), np.max(distance), np.min(distance))

    if mask is not None:
        distance = distance.astype(np.uint8)
        # disparity = cv2.cvtColor(disparity, cv2.COLOR_GRAY2BGR)
        # # print(disparity.shape, disparity.dtype)
        # # print(mask.shape, mask.dtype)
        # distance = add_mask(distance, mask)

    # height = imgL.shape[0]
    # width = imgL.shape[1]
    # imgL = Image.fromarray(imgL)
    # imgR = Image.fromarray(imgR)
    # imgD = Image.fromarray(disparity)

    # img_compare = Image.new('L', (width * 3, height))
    # img_compare.paste(imgL, box=(0, 0, width, height))
    # img_compare.paste(imgR, box=(width, 0, width * 2, height))
    # img_compare.paste(imgD, box=(width * 2, 0, width * 3, height))
    # res = np.array(img_compare)

    return distance


if __name__ == '__main__':

    from setting import *

    mask_r = cv2.imread(f"fig/NEWmask_r_164.png")
    mask_l = np.flip(mask_r, axis=1)
    h, w, _ = mask_r.shape

    fov = 150
    pupil_length = 0.06
    FocusLength = 0.4

    position = 0

    new_h = int(np.tan(fov / 360 * np.pi) / np.tan(164 / 360 * np.pi) * h)
    new_w = int(np.tan(fov / 360 * np.pi) / np.tan(164 / 360 * np.pi) * w)
    mask_r = mask_r[h // 2 - new_h // 2:h // 2 + new_h // 2, w // 2 - new_w // 2:w // 2 + new_w // 2]
    mask_l = mask_l[h // 2 - new_h // 2:h // 2 + new_h // 2, w // 2 - new_w // 2:w // 2 + new_w // 2]

    maskL_noBlind = cv2.resize(mask_l, (1860, 1860))
    maskR_noBlind = cv2.resize(mask_r, (1860, 1860))
    # maskL = maskL_noBlind
    # maskR = maskR_noBlind
    maskL = add_blind(copy.deepcopy(maskL_noBlind), side='left')
    maskR = add_blind(copy.deepcopy(maskR_noBlind), side='right')

    img_l = cv2.imread(f"fig_tmp/left_FOV{fov}-F{FocusLength}-pos{position}.jpg", cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(f"fig_tmp/right_FOV{fov}-F{FocusLength}-pos{position}.jpg", cv2.IMREAD_GRAYSCALE)
    # img_l = cv2.imread("fig_tmp/FOV150-F100-0EyeBall-left.jpg", cv2.IMREAD_GRAYSCALE)
    # img_r = cv2.imread("fig_tmp/FOV150-F100-0EyeBall-right.jpg", cv2.IMREAD_GRAYSCALE)

    img_l = blured_img(img_l)
    img_r = blured_img(img_r)
    img_l = add_mask(img_l, maskL)
    img_r = add_mask(img_r, maskR)

    h_fov = v_fov = fov
    hh = math.atan(pupil_length / 2 / FocusLength) * 180 / math.pi
    right_euler_angle = [0, hh, 0]
    left_euler_angle = [0, -hh, 0]

    maskL, maskR = epipolarCorrection(maskL, maskR, right_euler_angle,
                                      left_euler_angle, camera_focus_length, pupil_length,
                                      h_fov, v_fov)

    # maskL = cv2.cvtColor(maskL, cv2.COLOR_GRAY2BGR)
    # maskR = cv2.cvtColor(maskR, cv2.COLOR_GRAY2BGR)

    mask = cv2.bitwise_and(maskL, maskR)

    resL, resR = epipolarCorrection(img_l, img_r, right_euler_angle,
                                    left_euler_angle, camera_focus_length, pupil_length,
                                    h_fov, v_fov)

    # cv2.imwrite('tmp_l_ori.png', mask)
    # cv2.imwrite('tmp_r_ori.png', resR)
    tmp = add_mask(resR, mask)
    cv2.imwrite('tmp_l_ori.png', tmp)
    cv2.imwrite('tmp_r_ori.png', resR)

    depthImg = depthMap(resL, resR, mask)
    # cv2.imwrite('tmp_r_ori.png', depthImg)
    # depthImg = depthMap(resL, resR)
    # hc, wc = 930, 930
    # radius = 400
    # depthImg = depthImg[hc-radius:hc+radius, wc-radius:wc+radius]
    # depthImg = add_mask(depthImg, mask)
    cax = plt.imshow(depthImg, cmap='gray', vmin=0, vmax=255)
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar(cax, ticks=[0, 48, 96, 144, 192, 240])
    cbar.ax.set_yticklabels(['0 m', '3 m', '6 m', '9 m', '12 m', '15 m'])

    plt.show()
    plt.savefig(f'tmp{position}.png')

    # cv2.imwrite(f'fig/tmp{i}.png', depthImg)
    # print(depthImg)
