import cv2
import os
import numpy as np
import math
#the unified image size
IMG_WIDTH=1024
IMG_HEIGHT=768
CHANNELS=3
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
import tensorflow as tf

def Resize(img,width,height):
    """
        function:
            Resize(prj,width,height)
        goal:
            resize the picture to identified size,which I have defined in the the top of this class
        gras:
            prj-the image you want to resize.
        return:
            prj-the Mat of opencv2 picture type.
        """   
    img = cv2.resize(img,(width,height))
    return img

def SplitImage(img):
        """
        function:
            SplitImage(srcFileName,saveFileName)
        goal:
            This function writing to split a picture into two same part from the middle of this picture
        args:
            prj-the image you want to split
        return: 
            imgLeft-the left part of image
            imgRight-the right part of image
        """
        #get the width and hight of image
        width=img.shape[1]
        height=img.shape[0]
        #split the image
        imgLeft=img[0:height,0:int(width/2)]
        imgRight=img[0:height,int(width/2):]
        #return the left and right part of image
        return imgLeft,imgRight

def SplitImageByLine(img,offset):
        """
        function:
            SplitImage(srcFileName,saveFileName)
        goal:
            This function writing to split a picture into two same part from the middle of this picture
        args:
            prj-the image you want to split
            x-the width location of image,where you want to split the image
        return: 
            imgLeft-the left part of image
            imgRight-the right part of image
        """
        #get the width and hight of image
        width=img.shape[1]
        height=img.shape[0]
        x = int(offset)
        #split the image
        imgLeft=img[0:height, 0:x]
        imgRight=img[0:height, x:]
        #return the left and right part of image
        return imgLeft, imgRight

def CombineImages(img_left,img_right):
    """
    goal:
        combines two half images into one image.
    args:
        img_left-the left part of image.
        img_right-the right part of image.
    return:
        prj-the whole image.
    """
    img_left=cv2.resize(img_left,(int(IMG_WIDTH/2),IMG_HEIGHT))
    img_right=cv2.resize(img_right,(int(IMG_WIDTH/2),IMG_HEIGHT))
    #the height of new picture,which must be the largest height
    height=IMG_HEIGHT
    #the width of new picture,which is the sum of left and right part
    width=IMG_WIDTH
    #the channels of new picture
    channels=3
    #new a numpy array to store the new image    
    img=np.zeros((height,width,channels),dtype=np.uint8)
    img[0:height,0:int(width/2)]=img_left
    img[0:height,int(width/2):]=img_right
    return img


def DoubleImage(img):
    """
    goal:
        copy the image and combine the copy image and origin image,the width is two times as origin image.
    args:
        prj-the cv2 image
    return:
        newImg-the doubled image
    """
    if img is None:

        raise Exception("the image is None!")

    height,width,channel=img.shape

    #using numpy array build a new image

    newImg=np.zeros((height,width*2,channel),dtype=np.uint8)

    #copy the origin to the left and right side of newImg

    newImg[0:height,0:width]=img[0:height,0:width]

    newImg[0:height,width:]=img[0:height,0:width]
    
    #return new image
    
    return newImg

def findToLine(img, mid):
    rows, cols,channels=img.shape
    #using the exists piexl value to fill the gap
    lineLeft=0
    lineRight=0
    for i in range(mid, mid-100, -1):
        g=np.sum(img[0:rows, i-1]-img[0:rows, i])/(3*rows)
        if(g>0):
            lineLeft=i
            break

    for i in range(mid, mid+100, 1):
        g=np.sum(img[0:rows, i+1]-img[0:rows, i])/(3*rows)
        if(g>0):
            lineRight=i
            break

    cv2.line(img,pt1=(mid,0),pt2=(mid,rows),color=(0,0,255),thickness=2)
    cv2.line(img,pt1=(lineLeft,0),pt2=(lineLeft,rows),color=(0,255,255),thickness=2)
    cv2.line(img,pt1=(lineRight,0),pt2=(lineRight,rows),color=(0,255,255),thickness=2)
    # cv2.imshow("img", cv2.resize(img_bin,(1024,768)))
    cv2.imshow("img", cv2.resize(img,(1024,768)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return lineLeft, lineRight

def FillGap(img,lineLeft,lineRight):
    """
    goal:
        as a result of the combined image have a gap in the middle of the picture, so I write this function to fill the gap 
    args:
        prj-the input image
    return:
        the well picture
    """
    #get the width,height and the channels of image
    rows,cols,channels=img.shape
    gap=lineRight-lineLeft
    img[0:rows, lineLeft:cols-gap]=img[:,lineRight:cols]
    img=img[0:cols-gap]
    img = cv2.resize(img,(1024, 768))
    return img


def GetRegion(img,pt,width,height):
        """
        function:
            GetRegion(prj,pt,width,height)
        goal:
            get you interested region in picture
        args:
            prj-the input image
            pt-the left toppoint of interesting region's coordinate,pt[0] is width,pt[1] is height
            width-the interesting region's width
            height-the interesting region's height
        return:
            prj-the interesting area of this image
        """
        img=img[pt[1]:pt[1]+height,pt[0]:pt[0]+width]
        return img

def TailorTheCameraImg(self,filePath,savePath,pt,width,height):
        """
        goal:
            because the pictures which we get from camera are very large,so we suppose to get the interesting region and then resize it.
        args:
            filePath-the input image directory path.
            savePath-the save path.
            pt-the left toppoint of interesting region's coordinate,pt[0] is width,pt[1] is height.
            width-the interesting region's width.
            height-the interesting region's height.
        return:
            null
        """
        count=0
        #get all the images's name
        imgNames=os.listdir(filePath)
        #travels all the images and tailor them and save to save path
        for imgName in imgNames:
            if imgName.endswith(".jpg") or imgName.endswith(".png"):
                #read each image
                img=cv2.imread(filePath+imgName,cv2.IMREAD_UNCHANGED)
                #get the interesting area of this image
                img=GetRegion(img,pt,width,height)
                #save the file to the save path
                cv2.imwrite(savePath+imgName,img)
                count+=1
                print("Tailored and saved %d pictures!"%count)

def AlignImage(img,refImg):
    """
    goal:
        using homography matrix to correction image,we will get the homography matrix by ORB algorithm
    args:
        prj-the Mat image which we are going to correct.
        refImg-the reference image
    return:
        imgReg-the corrected image
        h-the homography matrix
    """
    #get the Gray scale images
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    refGray=cv2.cvtColor(refImg,cv2.COLOR_BGR2GRAY)
    #Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(imgGray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(refGray, None)
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    # Draw top matches
    imMatches = cv2.drawMatches(img, keypoints1, refImg, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = refImg.shape
    imgReg=np.zeros(img.shape,dtype=np.uint8)
    try:
        imgReg = cv2.warpPerspective(img, h, (width, height))
    except Exception:
        print("There is a bug in opencv's function")

    return imgReg, h

def AlignImageWithH(img,H):
    """
    goal:
        alignimg image with a given homography matrix.
    args:
        prj-the image we are going to correction.
        H-the homography matrix with the shape of (3,3)
    return:
        imgReg-the aligned image
    """
    height, width, channels=img.shape
    imgReg=np.zeros(img.shape, dtype=np.uint8)
    imgReg=cv2.warpPerspective(img, H, (width,height))
    return imgReg

def isNumber(str):
    n=len(str)
    for i in range(0, n-1):
        if str[i]<'0' or str[i]>'9':
            return False
    return True

def rename(filePath):
    """
    goal:
        renaming all files in the filePath directory,using 0 as the prefix,such as "1.jpg" would be "0001.jpg"
    args:
        filePath-the file path of files
    return:
        null
    """
    #rename all images
    fileNames=os.listdir(filePath)
    first_name=fileNames[0]
    if isNumber(first_name[:-4]):
        fileNames.sort(key=lambda x: int(x[:-4]))
    else:
        sorted(fileNames)

    for count, fileName in enumerate(fileNames):
        name,suffix=os.path.splitext(fileName)
        imgName=str(count)+".jpg"
        os.rename(os.path.join(filePath, fileName), os.path.join(filePath, imgName))
        print("Having renamed %d images"%count)

def makeImg(width,height,grayscale):
    """
    makeImg(width,height,grayscale) -> img
    brief :this function is used to make images.
    :param width: width
    :param height: height
    :param grayscale: the grayscale.
    :return: img
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img = np.full(img.shape, grayscale)
    cv2.imwrite("./surf.jpg", img)

def FID(imTag, imRef, flag=0):
    """
    FID(imTag , imRef, flag) -> fid
    :brief : computes the FID of two images
    :param imTag: the target image
    :param imRef: the reference image
    :param flag: if flag=0 compute it byu rows, else cmputes it by cols
    :return: fid
    """
    cvflage = None
    if (flag == 0):
        cvflage = cv2.COVAR_ROWS
    if (flag == 1):
        cvflage = cv2.COVAR_ROWS

    if imTag.size == 0 or imRef.size == 0:
        raise RuntimeError('image not exist, please check!')

    imTag = imTag / 255.
    imRef = imRef / 255.
    #computes mean and covariance
    covar1, mean1 = cv2.calcCovarMatrix(imTag[:, :, 0], mean=None, flags=cvflage | cv2.COVAR_NORMAL)
    for i in range(1, 3, 1):
        covar, mean = cv2.calcCovarMatrix(imTag[:, :, i], mean=None, flags=cvflage | cv2.COVAR_NORMAL)
        covar1 += covar
        mean1 += mean1
    covar1 /= 3
    mean1 /= 3

    covar2, mean2 = cv2.calcCovarMatrix(imRef[:, :, 0], mean=None, flags=cvflage | cv2.COVAR_NORMAL)
    for i in range(1, 3, 1):
        covar, mean = cv2.calcCovarMatrix(imRef[:, :, i], mean=None, flags=cvflage | cv2.COVAR_NORMAL)
        covar2 += covar
        mean2 += mean
    covar2 /= 3
    mean2 /= 3
    #computes fid
    mean = np.linalg.norm(mean2-mean1)
    covar = covar1+covar2 - 2*(np.sqrt(covar1*covar2))
    covar = np.trace(covar)
    fid = np.sqrt(covar+mean)
    print('fid : ',fid)
    return fid

def psnr(imTag, imRef):
    """
    psnr(imTag, imRef) -> psnr
    :brief : computes the psnr of two images.
    :param imTag: the taeget image
    :param imRef: the reference image
    :return: psnr
    """
    graph = tf.Graph()
    with graph.as_default():
        imTag = np.array(imTag, dtype=np.float32)
        imRef = np.array(imRef, dtype=np.float32)
        imTensor = tf.convert_to_tensor(imTag)
        refTensor = tf.convert_to_tensor(imRef)
        psnr = tf.image.psnr(imTensor, refTensor, 255)
        with tf.Session() as sess:
            psnr = psnr.eval()
    # print('psnr :', psnr)
    return psnr

def ssim(imTag, imRef):
    """
    :brief : compute the ssim f two images
    :param imTag:
    :param imRef:
    :return:
    """
    graph = tf.Graph()
    with graph.as_default():
        imTag = np.array(imTag, dtype=np.float32)
        imRef = np.array(imRef, dtype=np.float32)
        imTensor = tf.convert_to_tensor(imTag)
        refTensor = tf.convert_to_tensor(imRef)
        ssim = tf.image.ssim(imTensor, refTensor, 255)
        with tf.Session() as sess:
            ssim = ssim.eval()
    # print('ssim :', ssim)
    return ssim

def mse(imTag, imRef):
    """
    :brief : compute the mse f two images
    :param imTag:
    :param imRef:
    :return:
    """
    rmse = np.sqrt(np.mean(np.square(imTag-imRef)))
    # print('rmse :', rmse)
    return rmse

def process(title,now,total,startTm,nowTm):
    """
    @ brief : show the process status
    :param title: name of this process follow you want.
    :param now: the number of now
    :param total: total number
    :param startTm: start time
    :param nowTm: now time
    :return:
    """
    rate = (now/total)*100
    h = int((-startTm+nowTm)//3600)
    m = int(((-startTm+nowTm)%3600)//60)
    s = int((-startTm+nowTm)%60)
    print('{} : rate {:<2.2f}% time {:<2d}:{:<2d}:{:<2d} ['.format(title, rate, h, m, s), end='')
    for i in range(0, int((now/total)*100)):
        print('#', end='')
    for i in range(int((now/total)*100), 100):
        print(' ', end='')
    print(']\r')

def ROI(rg,flag):
    """
    @brief : compute the mean and variance of an image region
    :param rg: image region, cv.Mat type
    :return: mean : mean of this region
             var : variance of this region
    """
    # rg = np.array(rg, dtype=np.float32)
    #     # mean = np.mean(rg)
    #     # var = np.sqrt(np.var(rg))
    if flag==1:
        rg = cv2.cvtColor(rg, cv2.COLOR_RGB2GRAY)
        m, s = cv2.meanStdDev(rg)
        m = np.sum(m)
        s = np.sum(s)
        return m, s
    elif flag==0:
        m, s = cv2.meanStdDev(rg)
        m = np.sum(m)
        s = np.sum(s)
        return m, s

def getWindow(img):
    """
    @ brief : get the x,y,w,h and middle of camera capture images's center part.
    :param img: image
    :return: x,y,w,h and middle
    """
    rows, cols, c = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    retval, img_bin = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY)
    x_left=0
    x_right=0
    top=0
    bottom=0
    for i in range(1, cols, 1):
        g=np.sum(img_bin[0:rows, i]-img_bin[0:rows, i-1])/(rows*3)
        if g>= 1:
            x_left=i
            break

    for i in range(cols-2, 0, -1):
        g=np.sum(img_bin[0:rows, i]-img_bin[0:rows, i+1])/(rows*3)
        if g>= 1:
            x_right=i
            break

    for i in range(1,rows,1):
        g=np.sum(img_bin[i,0:cols]-img_bin[i-1,0:cols])/(cols*3)
        if g>0:
            top = i
            break

    for i in range(rows-2,0,-1):
        g=np.sum(img_bin[i,0:cols]-img_bin[i+1,0:cols])/(cols*3)
        if g>0:
            bottom = i
            break
    mid = (x_right+x_left)//2+50
    width = x_right - x_left
    height = bottom - top
    x = x_left
    y = top

    cv2.line(img,pt1=(mid,0),pt2=(mid,rows),color=(0,255,255),thickness=2)
    cv2.line(img,pt1=(x_left,0),pt2=(x_left,rows),color=(0,255,255),thickness=2)
    cv2.line(img,pt1=(x_right,0),pt2=(x_right,rows),color=(0,255,255),thickness=2)
    cv2.line(img,pt1=(0,top),pt2=(cols,top),color=(0,255,255),thickness=2)
    cv2.line(img,pt1=(0,bottom),pt2=(cols,bottom),color=(0,255,255),thickness=2)
    # cv2.imshow("img", cv2.resize(img_bin,(1024,768)))
    cv2.imshow("img", cv2.resize(img,(1024,768)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return x, y, width, height, mid

def join(path1,path2):
    if not path1.endswith('/'):
        path1 += '/'
    return path1+path2

def debugMsg(key,value,debug_flag):
    if debug_flag:
        print('debug message "{}" : , value={}'.format(key, value))

def debugImg(mesg,img,debug_flag):
    if debug_flag:
        title='dbmesg'+mesg
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__=="__main__":
    im1 = cv2.imread('resources/evaluation/Ours/21.JPG')
    getWindow(im1)