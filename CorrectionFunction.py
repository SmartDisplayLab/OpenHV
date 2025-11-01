import cv2
from numpy import array, zeros, where, max, min, dot, linalg, pi, sin, cos, tan
import numpy as np


def inner_matrix(camera_focus_length, img_h, img_w, camera_h_fov, camera_v_fov):
    """
    相机内参矩阵计算

    :param camera_focus_length: 相机焦距（毫米）
    :param img_h: 图像高度
    :param img_w: 图像宽度
    :param camera_h_fov: 相机水平视场角
    :param camera_v_fov: 相机垂直视场角
    :return: 3*3相机内参矩阵
    """
    realW = 2 * camera_focus_length * tan(pi * camera_h_fov / 360)
    realH = 2 * camera_focus_length * tan(pi * camera_v_fov / 360)
    res = zeros((3, 3), dtype=float)
    res[0, 0] = camera_focus_length * img_w / realW
    res[1, 1] = camera_focus_length * img_h / realH
    res[2, 2] = 1
    res[0, 2] = img_w / 2.
    res[1, 2] = img_h / 2.
    return res


def euler_angle2rotation_matrix(euler_angle, data_type='degree'):
    """
    以给定ZYX欧拉角计算旋转矩阵

    :param euler_angle: [x, y, z] 角度制ZXY欧拉角
    :param data_type: 角度制还是弧度制
    :return: 3*3旋转矩阵
    """
    if data_type == 'degree':
        euler_angle = [i * pi / 180.0 for i in euler_angle]

    R_x = array([[1, 0, 0],
                 [0, cos(euler_angle[0]), -sin(euler_angle[0])],
                 [0, sin(euler_angle[0]), cos(euler_angle[0])]
                 ])

    R_y = array([[cos(euler_angle[1]), 0, sin(euler_angle[1])],
                 [0, 1, 0],
                 [-sin(euler_angle[1]), 0, cos(euler_angle[1])]
                 ])

    R_z = array([[cos(euler_angle[2]), -sin(euler_angle[2]), 0],
                 [sin(euler_angle[2]), cos(euler_angle[2]), 0],
                 [0, 0, 1]
                 ])

    # Unity官方约定俗成的欧拉角层级关系是ZXY
    # 即最里层是Z轴先旋转，中间层是X轴，最外层是Y轴。
    # 后旋转矩阵左乘先旋转矩阵
    R = dot(R_y, dot(R_x, R_z))
    return R


def epipolarCorrection(img_l, img_r, right_euler_angle, left_euler_angle, camera_focus_length,
                       pupil_length, h_fov, v_fov):
    # if len(img_l.shape) == 3:
    #     img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    # if len(img_r.shape) == 3:
    #     img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    # 相机内参矩阵求解
    innerL = inner_matrix(camera_focus_length, img_l.shape[0], img_l.shape[1], h_fov, v_fov)
    innerR = inner_matrix(camera_focus_length, img_r.shape[0], img_r.shape[1], h_fov, v_fov)
    innertmp = innerL

    # 相机畸变矩阵：认为没有畸变（理想针孔模型）
    distortion = zeros((1, 5))

    # 相机旋转矩阵求解
    rotationL = euler_angle2rotation_matrix(left_euler_angle)
    rotationR = euler_angle2rotation_matrix(right_euler_angle)
    # print(rotationL, rotationR)

    # 相机平移矩阵求解（根据瞳距）
    T = array([1, 0, 0], dtype=np.float64)
    # print(T)

    # 极线矫正
    method = '2'
    # 计算矫正参数

    # 左相机旋转到右相机

    # rotation为相机之间的旋转矩阵
    # 这里rotation的意义是：左相机1通过变换rotation到达右相机2的位姿
    # ie. rotationR = rotation * rotationL
    if method == '1':
        rotation = dot(rotationR, linalg.inv(rotationL))

        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            innerL, distortion, innerR, distortion,
            (img_l.shape[1], img_l.shape[0]),
            R=rotation, T=T)

    # print(P1, '\n', P2, '\n', rotation)

    # 分别旋转到平行位置
    elif method == '2':
        R1, _, P1, _, Q, validPixROI1, ROI_LT = cv2.stereoRectify(
            innerL, distortion, innertmp, distortion,
            (img_l.shape[1], img_l.shape[0]),
            R=linalg.inv(rotationL), T=T)
        R2, _, P2, _, Q, validPixROI2, ROI_RT = cv2.stereoRectify(
            innerR, distortion, innertmp, distortion,
            (img_l.shape[1], img_l.shape[0]),
            R=linalg.inv(rotationR), T=T)
        # # print(rotationL)
        # # print(rotationR)
        # print(validPixROI1, ROI_LT, validPixROI2, ROI_RT)

    # 计算重投影参数
    map1_1, map1_2 = cv2.initUndistortRectifyMap(innerL, distortion, R1, P1, (img_l.shape[1], img_l.shape[0]),
                                                 cv2.CV_16SC2)
    map2_1, map2_2 = cv2.initUndistortRectifyMap(innerR, distortion, R2, P2, (img_r.shape[1], img_r.shape[0]),
                                                 cv2.CV_16SC2)

    # 左右眼图像重投影并获得矫正结果
    resultL = cv2.remap(img_l, map1_1, map1_2, cv2.INTER_LINEAR)
    resultR = cv2.remap(img_r, map2_1, map2_2, cv2.INTER_LINEAR)

    if len(img_l.shape) == 3:
        rows, cols, _ = resultR.shape
    else:
        rows, cols = resultR.shape

    if method == '2':
        # 平移矩阵M：[[1,0,x],[0,1,y]]
        shift = ROI_RT[0]
        M = np.float32([[1, 0, -shift], [0, 1, 0]])
        M_inv = np.float32([[1, 0, shift], [0, 1, 0]])
        resultR = cv2.warpAffine(resultR, M, (cols, rows))
        resultL = cv2.warpAffine(resultL, M_inv, (cols, rows))

    return resultL, resultR


if __name__ == '__main__':
    from setting import *
    from codeHV.ImageProcessFunction import *
    #
    img_l = cv2.imread(f"0EyeBall-left.jpg", cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(f"0EyeBall-right.jpg", cv2.IMREAD_GRAYSCALE)

    img_l = blured_img(img_l)
    img_r = blured_img(img_r)
    img_l = add_mask(img_l, maskL)
    img_r = add_mask(img_r, maskR)

    # Do correction
    resL, resR = epipolarCorrection(img_l, img_r, right_euler_angle,
                                    left_euler_angle, camera_focus_length, pupil_length,
                                    h_fov, v_fov)

    # # equal shift
    # validL = np.where(maskL > 0)
    # validR = np.where(maskR > 0)
    #
    # x_min_L = np.min(validL[1])
    # x_max_L = np.max(validL[1])
    # x_min_R = np.min(validR[1])
    # x_max_R = np.max(validR[1])
    #
    h, w = img_l.shape

    cv2.arrowedLine(resL, (w // 2, 0), (w // 2, h), color=(0, 0, 0))
    cv2.arrowedLine(resR, (w // 2, 0), (w // 2, h), color=(0, 0, 0))

    cv2.imwrite('tmp_l.png', resL)
    cv2.imwrite('tmp_r.png', resR)
