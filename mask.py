import cv2
import numpy as np
from ImageProcessFunction import *
import copy

mask_r = cv2.imread(f"fig/NEWmask_r_164.png")
mask_l = np.flip(mask_r, axis=1)
h, w, _ = mask_r.shape

fov = 164
new_h = int(np.tan(fov/360*np.pi) / np.tan(164/360*np.pi) * h)
new_w = int(np.tan(fov / 360 * np.pi) / np.tan(164 / 360 * np.pi) * w)
mask_r = mask_r[h//2-new_h//2:h//2+new_h//2, w//2-new_w//2:w//2+new_w//2]
mask_l = mask_l[h//2-new_h//2:h//2+new_h//2, w//2-new_w//2:w//2+new_w//2]

maskL_noBlind = cv2.resize(mask_l, (1860, 1860))
maskR_noBlind = cv2.resize(mask_r, (1860, 1860))
maskL = add_blind(copy.deepcopy(maskL_noBlind), side='left')
maskR = add_blind(copy.deepcopy(maskR_noBlind), side='right')

cv2.imwrite('tmp_l.png', maskL)
cv2.imwrite('tmp_r.png', maskR)
