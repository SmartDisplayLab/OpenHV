import numpy as np


def cat2images(limg, rimg):
    HEIGHT = limg.shape[0]
    WIDTH = limg.shape[1]
    imgcat = np.zeros((HEIGHT, WIDTH * 2 + 20))
    imgcat[:, :WIDTH] = limg
    imgcat[:, -WIDTH:] = rimg
    # for i in range(int(HEIGHT / 32)):
    #     imgcat[i * 32, :] = 255
    return imgcat


def cat3images(img1, img2, img3):
    HEIGHT = img1.shape[0]
    WIDTH = img1.shape[1]
    imgcat = np.zeros((HEIGHT, WIDTH * 3 + 20))
    imgcat[:, :WIDTH] = img1
    imgcat[:, WIDTH:2*WIDTH:] = img2
    imgcat[:, 2*WIDTH:3 * WIDTH] = img3

    # for i in range(int(HEIGHT / 32)):
    #     imgcat[i * 32, :] = 255
    return imgcat
