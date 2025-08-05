import numpy as np
import matplotlib.pyplot as plt
import cmath

with open('data.npy', 'rb') as f:
    t = np.load(f)
    r = np.load(f)

fov = 164
ind = np.where(r < 90)
t = t[ind]
r = r[ind]
# t = np.concatenate((t, np.zeros(1)), axis=None)
# r = np.concatenate((r, np.array([60])), axis=None)

# print(t.shape, r.shape)

r = np.tan(r/360*2*np.pi)

x = []
y = []

for ti, ri in zip(t, r):
    c = cmath.rect(ri, ti)
    x.append(c.real)
    y.append(c.imag)

x = np.array(x)
y = np.array(y)

ind_g = np.where(y >= 0)
ind_l = np.where(y < 0)

x = np.concatenate((np.flip(x[ind_g]), np.flip(x[ind_l])), axis=None)
y = np.concatenate((np.flip(y[ind_g]), np.flip(y[ind_l])), axis=None)

fig, ax = plt.subplots()
ax.plot(-x, y, alpha=0.5)
# ax.plot(-x, y, alpha=0.5)
t_lim = np.tan(fov/360*np.pi)
ax.set_xlim([-t_lim, t_lim])
ax.set_ylim([-t_lim, t_lim])
ax.set_aspect('equal', adjustable='box')
# ax.grid(True)
plt.axis('off')
# ax.set_title("A line plot on a polar axis", va='bottom')
# plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# plt.margins(0, 0)
plt.savefig('fig/mask.png', bbox_inches='tight', pad_inches=0)
