import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
args = dict(
    width=1024,
    height=768,
    input_img_path=r"C:\canary\data\PaperData\input\23.jpg",
    img_paths=[
        r"C:\canary\data\PaperData\hemispherical_corrected\uncorrected\23.jpg",
        r"C:\canary\data\PaperData\hemispherical_corrected\singlenet\gamma\gamma=0.8\23.jpg",
        r"C:\canary\data\PaperData\hemispherical_corrected\singlenet\gamma\gamma=1.0\23.jpg",
        r"C:\canary\data\PaperData\hemispherical_corrected\singlenet\gamma\gamma=1.4\23.jpg",
        r"C:\canary\data\PaperData\hemispherical_corrected\singlenet\gamma\gamma=1.8\23.jpg",
        r"C:\canary\data\PaperData\hemispherical_corrected\singlenet\gamma\gamma=2.2\23.jpg",
    ],
    name_lists=[
        "uncorrected",
        "gamma=2.2",
        "gamma=1.0",
        "gamma=0.71",
        "gamma=0.56",
        "gamma=0.45"
    ]
)

points = []
for i in range(0, 6):
    points.append([(args['width'] // 6) * i + (args['width'] // 12), args['height'] // 2])
uncorrected_points = []
for i in range(0, 6):
    uncorrected_points.append([(args['width'] // 6) * i + (args['width'] // 12), args['height'] // 2])

def ColorDif(src_px, prj_px):
    return math.sqrt(2 * math.pow(src_px[0] - prj_px[0], 2)
                     + 4 * math.pow(src_px[1] - prj_px[1], 2)
                     + 3 * math.pow(src_px[2] - prj_px[2], 2))

def GetDiffList(src_img, prj_img, file=None, name=""):
    src_img = np.array(src_img, dtype=np.float32)
    prj_img = np.array(prj_img, dtype=np.float32)
    diffs = []
    for point, ucpoint in zip(points, uncorrected_points):
        pt1 = src_img[point[1], point[0]]
        pt2 = prj_img[ucpoint[1], ucpoint[0]]
        diffs.append(ColorDif(pt1, pt2))
    return diffs

def GetAllDiffs():
    diffs = []
    input = cv2.resize(cv2.imread(args['input_img_path']), (args['width'], args['height']))
    for path, name in zip(args['img_paths'], args['name_lists']):
        img = cv2.resize(cv2.imread(path), (args['width'], args['height']))

        diffs.append(np.array(GetDiffList(input, img, None, name)))
    return diffs

diffs = GetAllDiffs()
x = np.arange(6)
# 有多少个类型，只需更改n即可
total_width, n = 0.8, len(diffs)
width = total_width/(2*n)
# 重新拟定x的坐标
x = x - (total_width/2 - width) / 2
# 这里使用的是偏移
for i in range(0, n):
    plt.bar(x+i*width, diffs[i], width=width, label=args['name_lists'][i])
plt.legend()
plt.savefig("./color_diffs.jpg")
plt.close()
