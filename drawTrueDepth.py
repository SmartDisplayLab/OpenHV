import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ImageProcessFunction import *
from setting import *
import matplotlib.colors as mcolors

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
maskL = add_blind(copy.deepcopy(maskL_noBlind), side='left')
maskR = add_blind(copy.deepcopy(maskR_noBlind), side='right')

img_l = cv2.imread(f"fig_tmp/Dleft_FOV{fov}-F{FocusLength}-pos{position}.jpg", cv2.IMREAD_GRAYSCALE)
img_r = cv2.imread(f"fig_tmp/Dright_FOV{fov}-F{FocusLength}-pos{position}.jpg", cv2.IMREAD_GRAYSCALE)

h_fov = v_fov = fov
hh = math.atan(pupil_length / 2 / FocusLength) * 180 / math.pi
right_euler_angle = [0, hh, 0]
left_euler_angle = [0, -hh, 0]

maskL, maskR = epipolarCorrection(maskL, maskR, right_euler_angle,
                                  left_euler_angle, camera_focus_length, pupil_length,
                                  h_fov, v_fov)

resL, resR = epipolarCorrection(img_l, img_r, right_euler_angle,
                                left_euler_angle, camera_focus_length, pupil_length,
                                h_fov, v_fov)

# plt.hist(np.reshape(img_l, -1), bins=30)
# plt.show()
# plt.hist(np.reshape(resL, -1), bins=30)
# plt.show()

mask = cv2.bitwise_and(maskL, maskR)
depthImg = 255 - add_mask(255 - resL, mask)
colors = ['black', 'white']
cmap = mcolors.LinearSegmentedColormap.from_list('cmap1', colors)
cax = plt.imshow(depthImg, cmap=cmap, vmin=0, vmax=255)

cbar = plt.colorbar(cax, ticks=[0, 48, 96, 144, 192, 240])
cbar.ax.set_yticklabels(['0 m', '3 m', '6 m', '9 m', '12 m', '15 m'])


plt.xticks([])
plt.yticks([])


plt.savefig(f'GTDtmp{position}.png')
