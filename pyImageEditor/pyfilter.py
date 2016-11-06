import numpy as np
import scipy.ndimage
import matplotlib.image as mpimg

class pyfilter:
    def __init__(self, img_directory):
        self.img = mpimg.imread(img_directory)
        self.grey = False

    def getImage(self):
        return self.img

    def set_image(self, img):
        self.img = img

    def rgb2gray(self, rgb):
        self.grey = True
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    def gaussKernel(self, sz=5, sigma=1):
        kernel = np.zeros((sz,sz))
        x = - sz // 2 + 1
        add = sz //2

        while x <= sz // 2:
            y = - sz // 2 + 1
            while y <= sz // 2:
                kernel[x + sz//2][y + sz//2] = np.exp(-(x**2 + y**2) / float(2 * sigma **2))
                y += 1
            x += 1
        kernel /= sum(sum(kernel))

        self.GK = kernel
        return self.GK

    def meanFilt(self, sz = 3):
        x = np.ones((sz, sz))
        self.AB = x / sum(sum(x))
        return self.AB

    def applyFilter(self, img, fltr):
        return scipy.ndimage.convolve(img, fltr)

    def gaussianSmooth(self):
        self.gausKernel()
        img2R_gaus = self.applyFilter(self.img[:, :, 0], self.GK)
        img2G_gaus = self.applyFilter(self.img[:, :, 1], self.GK)
        img2B_gaus = self.applyFilter(self.img[:, :, 2], self.GK)
        return np.dstack((img2R_gaus, img2G_gaus, img2B_gaus))

    def meanFilter(self):
        self.meanFilt()
        img2R_mf = self.applyFilter(self.img[:, :, 0], self.AB)
        img2G_mf = self.applyFilter(self.img[:, :, 1], self.AB)
        img2B_mf = self.applyFilter(self.img[:, :, 2], self.AB)
        return np.dstack((img2R_mf, img2G_mf, img2B_mf))
