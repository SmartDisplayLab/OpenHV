import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import Rbf, griddata
from matplotlib.colors import LogNorm
import brewer2mpl
import seaborn as sns

filename = 'retinal topography calculator data1.xlsx'

data = pd.read_excel(filename)

data = data.filter(items=['x', 'y', 'density'])

x = data['x'].to_numpy()
y = data['y'].to_numpy()
d = data['density'].to_numpy()

# d = np.log10(d)

x = x[:, np.newaxis]
y = y[:, np.newaxis]

points = np.hstack((x, y))
l = 90
grid_x, grid_y = np.mgrid[-l:l, -l:l]

# get Nuiquist freq
F = np.sqrt(d / (2 * np.sqrt(3)))
# get standard visual acuity
VA = F * 2 / 60

print(VA.max(), VA.min())

grid_d = griddata(points, d, (grid_x, grid_y), method='linear')
densityList = [0, 1, 4, 10, 100, 1000, 5000, 20000]

# densityList = [0, 0.02, 0.04, 0.1, 0.2, 0.4, 1.2, 2.4]
# grid_d = griddata(points, VA, (grid_x, grid_y), method='linear')

# densityList = [0, 0.5, 1, 2, 4, 8, 16, 32, 64]
# plt.plot(points[:, 0], points[:, 1], 'k.', ms=1, color='black')
# 标准视力表

fig, ax = plt.subplots()
# current_palette = sns.cubehelix_palette(8, start=.5, rot=-.75)
# current_palette = sns.color_palette("hls", 7)[::-1]
current_palette = sns.light_palette("black", 8)
# current_palette = sns.cubehelix_palette(n_colors=8, rot=-0.2, start=1.2)
plt.contourf(grid_x, grid_y, grid_d, densityList,
             norm=LogNorm(), colors=current_palette, alpha=1)
cbar = plt.colorbar()
# cbar.set_label("Visual acuity (decimal)", rotation=270, labelpad=20)
cbar.set_label("Density of midget cells (/degree$^2$)", rotation=270, labelpad=20)
contour = plt.contour(grid_x, grid_y, grid_d, densityList,
                      norm=LogNorm(), colors=current_palette, alpha=1)

plt.xlabel('Horizontal degree')
plt.ylabel('Vertical degree')
# plt.xlim(-20, 20)
# plt.ylim(-20, 20)

# plt.clabel(contour, fontsize=6, colors='red')

# plt.imshow(grid_d.T, extent=(-100, 100, -100, 100), origin='lower')

# rbf = Rbf(x, y, d, function='linear', episilon=2)
#
# xi = np.linspace(-100, 100, 100)
# yi = np.linspace(-100, 100, 100)
# di = rbf(xi, yi)
#
# fig = plt.figure(figsize=(10, 8))
# ax = Axes3D(fig)
#
# ax.scatter(xi, yi, di)

# ax.plot_surface(X, Y, density,
#     rstride=1,  # rstride（row）指定行的跨度
#     cstride=1,  # cstride(column)指定列的跨度
#     cmap=plt.get_cmap('rainbow'))  # 设置颜色映射
# plt.xlabel('X轴', fontsize=15)
# plt.ylabel('Y轴', fontsize=15)
# ax.set_zlabel('Z轴', fontsize=15)
# ax.set_title('《曲面图》', y=1.02, fontsize=25, color='gold')
# 设置Z轴范围
# ax.set_zlim(-2, 2)
plt.show()
fig.savefig("density.svg", dpi=300, format='svg')

