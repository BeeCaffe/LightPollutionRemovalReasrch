from src.wrapnet.MyTrainUtils import *
import time
import src.wrapnet.utils
import src.wrapnet.CompenNetPlusplusModel as CompenNetPlusplusModel
from time import localtime,strftime
import numpy as np
__DEBUG=False
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() >= 1:
    print('Valid with', torch.cuda.device_count(), 'GPU...')
else:
    print('Valid with CPU...')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = {
    "desire_img_path": 'datasets/geocor/cam/train/',#images path of which you suppose to compensate
    "save_path": 'data/geocor/cmp/res',#directory of save compensated images
    'pth_path': 'checkpoint/Warpping-Net-1920-1080_l1+l2+ssim_50_4_100.pth',
    "size": (1920, 1080),#geometric corrected image size
}

def cmpImages(pth_path, desire_test_path, save_path):
    if not os.path.exists(pth_path):
        print("error : xxx.pth path not exist\n")
    warpnet_model = torch.load(pth_path)
    if isinstance(warpnet_model, torch.nn.DataParallel):
        warpnet_model = warpnet_model.module
    # warpnet_model = warpnet_model.warping_net
    torch.cuda.empty_cache()
    desire_test_path = desire_test_path
    # assert os.path.isdir(desire_test_path), 'images and folder {:s} does not exist!'.format(desire_test_path)
    prj_cmp_path = join(save_path, os.path.split(pth_path)[1][:-4])
    if not os.path.exists(prj_cmp_path):
        os.makedirs(prj_cmp_path)
    if not os.path.exists(prj_cmp_path): os.makedirs(prj_cmp_path)
    st = time.time()
    for i in range(len(os.listdir(desire_test_path))):
        desire_test = readImgsMT(desire_test_path, index=[i], size=args.get('size')).to(device)
        with torch.no_grad():
            warpnet_model.eval()
            prj_cmp_test = warpnet_model(desire_test).detach()  # compensated prj input image x^{*}
            saveImg(prj_cmp_test, prj_cmp_path, i)  # compensated testing images, i.e., to be projected to the surface
        del desire_test
        nt = time.time()
        process("compensating image...", i, len(os.listdir(desire_test_path)), st, nt)

def cmpImage(pth_path,imgs,size=args["size"]):
    if not os.path.exists(pth_path):
        print("error : xxx.pth path not exist\n")
    warpnet_model = torch.load(pth_path)
    if isinstance(warpnet_model, torch.nn.DataParallel):
        warpnet_model = warpnet_model.module
    torch.cuda.empty_cache()
    ans = []
    for img in imgs:
        if img is None:
            img = np.ones([size[0], size[1], 3], np.uint8)
        img = cv.resize(img, size)
        img = cv2Tensor(img).to(device)
        with torch.no_grad():
            warpnet_model.eval()
            img = warpnet_model(img).detach()  # compensated prj input image x^{*}
        img = tensor2Cv(img)
        img = cv.resize(img, size)
        ans.append(img)
    return ans

if __name__=="__main__":
    cmpImages(pth_path=args.get('pth_path'), desire_test_path=args.get('desire_img_path'), save_path=args.get('save_path'))
