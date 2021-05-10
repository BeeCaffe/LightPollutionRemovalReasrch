import cupy as np
import os
import cv2
import time
from src.tools.Utils import process
import scipy as sp
import numpy

class TPS():
    def __init__(self, input_img_dir=None, size=(256, 256)):
        self.camera_img_dir = r'F:\yandl\LightPollutionRemovalReasrch\input\tps\camIm/'
        self.projector_img_dir = r'F:\yandl\LightPollutionRemovalReasrch\input\tps/prjIm/'
        self.weight_save_dir = 'F:/tps_weight/'
        self.input_img_dir = input_img_dir
        self.save_img_dir = r'F:\yandl\LightPollutionRemovalReasrch\output\tps/'
        self.size = size
        self.cam_img_lists = None
        self.prj_img_lists = None
        self.N = 0
        self.loadImgs()
        self.out_size = (1920, 1080)

    def loadImgs(self):
        self.cam_img_lists = []
        self.prj_img_lists = []

        cam_img_paths = os.listdir(self.camera_img_dir)
        cam_img_paths.sort(key = lambda x:int(x[:-4]))
        prj_img_paths = os.listdir(self.projector_img_dir)
        prj_img_paths.sort(key = lambda x:int(x[:-4]))

        for name in cam_img_paths:
            path = self.camera_img_dir+name
            img = np.array(cv2.resize(cv2.imread(path), self.size)) / 255.
            self.cam_img_lists.append(img)

        for name in prj_img_paths:
            path = self.projector_img_dir+name
            img = np.array(cv2.resize(cv2.imread(path), self.size)) / 255.
            self.prj_img_lists.append(img)
        self.N = len(self.cam_img_lists)

    def loadQ(self, row, col):
        Q = np.zeros((self.N, 4), dtype=np.float)
        for i in range(self.N):
            cam_img = self.cam_img_lists[i]
            Q[i, 0] = 1.
            Q[i, 1:] = cam_img[row][col]
        return Q

    def loadP(self, row, col):
        P = np.zeros((self.N, 3), dtype=np.float)
        for i in range(self.N):
            prj_img = self.prj_img_lists[i]
            P[i, :] = prj_img[row][col]
        return P

    def loadK(self, Q):
        K = np.zeros((self.N, self.N), dtype=np.float)
        for i in range(self.N):
            for j in range(self.N):
                kij = 0.
                for k in range(4):
                    kij += np.power((Q[i, k] - Q[j, k]), 2)
                kij = self.fi(np.sqrt(kij))
                K[i][j] = kij
        return K

    def loadL(self, Q, K):
        L = np.zeros((self.N + 4, self.N + 4), dtype=np.float)
        Qt = Q.T
        O = np.zeros((4, 4), dtype=np.float)
        L[0:self.N, 0:self.N] = K
        L[0:self.N, self.N:] = Q
        L[self.N:, 0:self.N] = Qt
        L[self.N:, self.N:] = O
        return L

    def computeW(self, L, P):
        a = L
        b = np.ones((self.N + 4, 3), dtype=np.float)
        b[0:self.N, :] = P
        w = np.zeros((self.N+4, 3))
        try:
            # w = np.linalg.solve(a, b)
            w = sp.linalg.solve(a, b)
        except:
            print('do not exist solve！')
        return w

    def computeAllW(self, start_rows):
        st = time.time()
        for i in range(start_rows, self.size[0]):
            for j in range(self.size[1]):
                Q = self.loadQ(i, j)
                K = self.loadK(Q)
                L = self.loadL(Q, K)
                P = self.loadP(i, j)
                W = self.computeW(L, P)
                file_name = self.weight_save_dir+'w_'+str(i)+'x'+str(j)+'.txt'
                W = np.asnumpy(W)
                numpy.savetxt(file_name, W)
                et = time.time()
                process("compute w:", i*self.size[1]+j, self.size[0]*self.size[1], st, et)

    def TestSpeed(self, weight_mat, img):
        img = np.array(img)
        img[:, :, 0] = self.TestSgPxSpeed(img[:, :, 0], weight_mat)
        img[:, :, 1] = self.TestSgPxSpeed(img[:, :, 1], weight_mat)
        img[:, :, 2] = self.TestSgPxSpeed(img[:, :, 2], weight_mat)


    def TestSgPxSpeed(self, pixel, W):
        Q = self.loadQ(0, 0)
        b = 0.
        g = 0.
        r = 0.
        N = self.N
        for i in range(self.N):
            sum = 0.
            for j in range(3):
                sum += np.power(pixel[j] - Q[i, j + 1], 3)
            sum = self.fi(np.sqrt(sum))
            b += W[i, 0] * sum
            g += W[i, 1] * sum
            r += W[i, 2] * sum
        b += W[N, 0] + W[N + 1, 0] * pixel[2] + W[N + 2, 0] * pixel[1] + W[N + 3, 0] * pixel[0]
        g += W[N, 1] + W[N + 1, 1] * pixel[2] + W[N + 2, 1] * pixel[1] + W[N + 3, 1] * pixel[0]
        r += W[N, 2] + W[N + 1, 2] * pixel[2] + W[N + 2, 2] * pixel[1] + W[N + 3, 2] * pixel[0]
        pixel[0] = b
        pixel[1] = g
        pixel[2] = r
        return pixel

    def compenSgPx(self, pixel, row, col):
        w_file_name = self.weight_save_dir+'w_'+str(row)+'x'+str(col)+'.txt'
        W = numpy.loadtxt(w_file_name)
        Q = self.loadQ(row, col)
        b=0.
        g=0.
        r=0.
        N=self.N
        for i in range(self.N):
            sum = 0.
            for j in range(3):
                sum+=np.power(pixel[j] - Q[i, j + 1], 3)
            sum = self.fi(np.sqrt(sum))
            b += W[i, 0]*sum
            g += W[i, 1]*sum
            r += W[i, 2]*sum
        b += W[N, 0] + W[N + 1, 0] * pixel[2] + W[N + 2, 0] * pixel[1] + W[N + 3, 0] * pixel[0]
        g += W[N, 1] + W[N + 1, 1] * pixel[2] + W[N + 2, 1] * pixel[1] + W[N + 3, 1] * pixel[0]
        r += W[N, 2] + W[N + 1, 2] * pixel[2] + W[N + 2, 2] * pixel[1] + W[N + 3, 2] * pixel[0]
        pixel[0] = b
        pixel[1] = g
        pixel[2] = r
        return pixel

    def compenSgIm(self, img):
        st = time.time()
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                pixel = img[i, j, :]
                pixel = self.compenSgPx(pixel, i, j)
                img[i, j] = pixel
                et = time.time()
                # process("compen single pixel: ", i*self.size[1]+j, self.size[0]*self.size[1], st, et)
        return img

    def compenImgs(self):
        name_lists = os.listdir(self.input_img_dir)
        i = 0
        for name in name_lists:
            path = self.input_img_dir+name
            img = cv2.resize(cv2.imread(path), self.size)
            img = np.array(img / 255., dtype=np.float)
            img = self.compenSgIm(img)
            img = np.array(np.clip(img, a_min=0, a_max=1) * 255., dtype=np.uint8)
            # img = cv2.resize(img, self.out_size)
            save_path = self.save_img_dir+str(i)+'.jpg'
            i+=1
            cv2.imwrite(save_path, img)
        print('Done!')

    def fi(self, d):
        if (d< 1e-3).any():
            return 0.
        else:
            return np.power(d, 2) * np.log(d)

if __name__=='__main__':
    tps = TPS()
    tps.computeAllW(0)
    # tps.compenImgs()





