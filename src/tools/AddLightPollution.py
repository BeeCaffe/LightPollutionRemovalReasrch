import cv2
import math
import numpy as np


class ImageEnhance:
    def __init__(self, centerX, centerY, img, illumination):
        """
        :param centerX: 相对于图像左上角的X坐标比例,0<=centerX<=1
        :param centerY:  相对于图像左上角的Y坐标,0<=centerY<=1
        :param img: 图像路径
        :param illumination: 增加的光照值
        """
        self.src_img = cv2.imread(img)
        self.centerX = np.around(centerX * self.src_img.shape[0])
        self.centerY = np.around(centerY * self.src_img.shape[1])
        self.diagonal_distance = np
        self.illumination = illumination
        print('img shape:%s' % str(self.src_img.shape))

    def increase_illumination(self):
        rows, cols = self.src_img.shape[:2]
        dst_image = np.zeros(self.src_img.shape, dtype='uint8')
        diagonal_dis = self.src_img.shape[0] ** 2 + self.src_img.shape[1] ** 2  # 对角线距离平方
        for row in range(rows):
            for col in range(cols):
                distance = np.abs(row - self.centerX) ** 2 + np.abs(col - self.centerY) ** 2
                t_illumination = (1 - distance / diagonal_dis) * self.illumination
                if row == 0 and col == 0:
                    print(t_illumination)
                B = min(255, self.src_img[row, col][0] + t_illumination)
                G = min(255, self.src_img[row, col][1] + t_illumination)
                R = min(255, self.src_img[row, col][2] + t_illumination)
                dst_image[row, col] = np.uint8((B, G, R))
        return dst_image


temp = ImageEnhance(0.9, 0.9, 'image_set/interesting.png', 120)
dst = temp.increase_illumination()
src = cv2.imread('image_set/interesting.png ')
cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
