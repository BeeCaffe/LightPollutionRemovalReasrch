import cupy as np
import cv2 as cv
import os
import math
class Bimber:
    def __init__(self, imRoot, save_dir='output/bimber/'):
        self.width = 256
        self.height = 256
        self.screenWidth = 0.47
        self.screenHeight = 0.18
        self.screenMidHeight = 0.255
        self.centerDist = 0.55
        self.patchSize = 16
        self.imRoot = imRoot
        self.save_dir = save_dir
        self.u = self.width//self.patchSize
        self.v = self.height//self.patchSize
        self.detaX = self.screenWidth/self.u
        self.detaYEd = self.screenHeight/self.v
        self.detaYEd = self.screenMidHeight/self.v
        self.PI = 3.14159265358979323846
        self.f = 1
        self.onePathchF()
        self.doublePatchF()
        self.save_size = (1920, 1080)

    def onePathchF(self):
        self.Fi = np.zeros((self.v, self.u), np.float)
        for i in range(self.u*self.v):
            xi = (i % self.u)*self.detaX+self.detaX/2.
            yi = self.screenHeight - (i/self.u)*self.detaYEd/2.
            d = np.sqrt(-(1.1 * (self.screenWidth / 2. - xi) * 0.5253 - (self.screenWidth / 2. - xi) * (self.screenWidth / 2. - xi) - math.pow(.55, 2)))
            ri = np.sqrt(d * d + yi * yi)
            alpha_i = self.PI / 2 - math.atan(d / yi)
            dA = 1. / (self.u * self.v)
            ftmp = dA * math.cos(alpha_i) / (ri * ri * self.PI)
            col = int(i % self.u)
            row = i // self.u
            self.Fi[row, col] = self.f * ftmp
        Fi = np.zeros((self.width, self.height), dtype=np.float)
        for i in range(self.u):
            for j in range(self.v):
                Fi[i*self.patchSize:(i+1)*self.patchSize, j*self.patchSize:(j+1)*self.patchSize] = self.Fi[i][j]
        self.Fi = np.expand_dims(Fi, axis=2)

    def doublePatchF(self):
        self.Fij = np.zeros((self.u * self.v, self.u * self.v), dtype=np.float32)
        for i in range(self.u*self.v):
            for j in range(self.u*self.v):
                if (i % self.u < self.u / 2) and (j % self.u > self.u / 2):
                    di = self.screenWidth / 2-(i % self.u) * self.detaX-self.detaX / 2.
                    dj = (j % self.u-self.u / 2) * self.detaX+self.detaX / 2
                    rij = math.sqrt(di * di+dj * dj)
                    beta_i = math.atan(dj / di)
                    beta_j = math.atan(di / dj)
                    alpha_i = self.PI / 2-beta_i
                    alpha_j = self.PI / 2-beta_j
                    dA = 1. / (self.u * self.v)
                    ftmp = dA * math.cos(alpha_i) * math.cos(alpha_j) / (rij * rij * self.PI)
                    self.Fij[i, j] = ftmp
                elif (i % self.u >= self.u / 2) and (j % self.u <= self.u / 2):
                    dj=self.screenWidth / 2-(j % self.u) * self.detaX-self.detaX / 2.
                    di=(i % self.u-self.u / 2) * self.detaX+self.detaX / 2
                    rij=math.sqrt(di * di+dj * dj)
                    beta_i=math.atan(dj / di)
                    beta_j=math.atan(di / dj)
                    alpha_i=self.PI / 2-beta_i
                    alpha_j=self.PI / 2-beta_j
                    dA=1. / (self.u *self.v)
                    ftmp=dA * math.cos(alpha_i) * math.cos(alpha_j) / (rij * rij * self.PI)
                    self.Fij[i, j]=ftmp
        for i in range(self.u*self.v):
            sm = np.sum(self.Fij[i, :])
            self.Fij[i, :] /= sm

    def computeScatter(self, I):
        S = np.zeros((self.height, self.width, 3), dtype=np.float32)
        for i in range(0, self.u*self.v):
            row = i//self.u
            col = i%self.u
            patch = I[row*self.patchSize:(row+1)*self.patchSize, col*self.patchSize:(col+1)*self.patchSize]
            for j in range(0, self.u*self.v):
                S[row*self.patchSize:(row+1)*self.patchSize, col*self.patchSize:(col+1)*self.patchSize] += \
                    patch*self.Fi[row*self.patchSize, col*self.patchSize, 0]*self.Fij[j, i]
        S[np.where(S<0)] = 0
        return S

    '''
    R: the initial image
    S: the compensated image
    '''
    def CompensateI(self, R, S):
        I = np.subtract(R, S)
        I = np.divide(I, self.Fi)
        I[np.where(I<0)] = 0
        return np.array(I)


    def compensateImg(self, R):
        S = self.computeScatter(R)
        I = self.CompensateI(R, S)
        S_next = self.computeScatter(I)
        I = R - 0.15*S_next
        mid_w = self.v//2
        block = I[:, (mid_w+1)*self.patchSize:(mid_w+2)*self.patchSize]
        I[:, (mid_w+1)*self.patchSize:(mid_w+2)*self.patchSize] = np.power(block, 1.02)
        return np.array(I)

    def compensateImgs(self):
        nameLists = os.listdir(self.imRoot)
        for name in nameLists:
            imgPath = self.imRoot+"/"+name
            img = cv.resize(cv.imread(imgPath), (self.width, self.height))
            img = np.array(img, dtype=np.float32)
            n_img = self.compensateImg(img)
            n_img = np.asnumpy(n_img)
            n_img = cv.resize(n_img, self.save_size)
            saveName = self.save_dir+name
            cv.imwrite(filename=saveName, img=n_img)
            # print("compensated an image")
        # print('Done!')

if __name__=='__main__':
    bimber = Bimber(imRoot=r"input/pairnet/")
    bimber.compensateImgs()