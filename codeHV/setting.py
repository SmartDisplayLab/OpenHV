import cv2
from codeHV.ImageProcessFunction import add_blind
import copy
import numpy as np
import math

show_size = (480, 480)

# mask_r = cv2.imread(f"fig/mask_r.png")
# mask_l = np.flip(mask_r, axis=1)
# maskL_noBlind = cv2.resize(mask_l, (1860, 1860))
# maskR_noBlind = cv2.resize(mask_r, (1860, 1860))
# maskL = add_blind(copy.deepcopy(maskL_noBlind), side='left')
# maskR = add_blind(copy.deepcopy(maskR_noBlind), side='right')

# choiceDict = dict()
# choiceDict[100] = 0
# choiceDict[10] = 1
# choiceDict[1] = 2
# choiceDict['INF'] = 3
# choiceDict['0.4'] = 4
# choiceDict['0.2'] = 5
# choiceF = '0.2'
# choiceInd = choiceDict[choiceF]

# Focus 100, 10, 1 (0 1 2)
# right_euler_angle_all = [[0, 0.017, 0], [0, 0.172, 0], [0, 1.718, 0], [0, 0, 0], [0, 4.289, 0], [0, 8.531, 0]]
# left_euler_angle_all = [[0, -0.017, 0], [0, -0.172, 0], [0, -1.718, 0], [0, 0, 0], [0, -4.289, 0], [0, -8.531, 0]]
# right_euler_angle = right_euler_angle_all[choiceInd]
# left_euler_angle = left_euler_angle_all[choiceInd]

camera_focus_length = 24.91928
# pupil_length = 0.6
# h_fov = 150
# v_fov = 150
