from  src.compennetplusplus.trainNetwork import *
import src.compennetplusplus.myutils as myutils
import time
import src.compennetplusplus.CompenNetPlusplusModel
from time import localtime,strftime
import os
import torch
import cv2 as cv
__DEBUG=False
os.environ['CUDA_VISIBLE_DEVICES']='0'
device_ids=[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()>=1:
    print('Valid with',torch.cuda.device_count(),'GPU...')
else:
    print('Valid with CPU...')
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

args = {
    "desire_img_path": 'data/dataset_256_gama1.5/cmp/desire',#images path of which you suppose to compensate
    "save_path": 'H:\compensated goodimge\compenNet++\\',#directory of save compensated images
    'pth_path': 'I:\Backup\\2019.10.23\compenNet++_res\CompenNet++_l1+ssim_4200_1_5_0.001_0.2_5000_0.0001.pth',
    'cmp_surf_path': 'data/dataset_256_gama1.5/cmp/surf/surf.jpg',
    "size": (1080, 1920),#compensated image's size
}

def cmpImages(pth_path,desire_test_path,cmp_surf_path,save_path):
    if not os.path.exists(pth_path):
        print("error : xxx.pth path not exist\n")
    model = torch.load(pth_path)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    # compen_model = model.compen_net
    compen_model = model
    torch.cuda.empty_cache()
    desire_test_path = desire_test_path
    assert os.path.isdir(desire_test_path), 'images and folder {:s} does not exist!'.format(desire_test_path)
    cmp_surf = cv.imread(cmp_surf_path, cv.IMREAD_UNCHANGED)
    cmp_surf = cv.resize(cmp_surf, (args['size'][1], args['size'][0]))
    cmp_surf = cv.cvtColor(cmp_surf, cv.COLOR_BGR2RGB)[np.newaxis, :, :, :]
    cmp_surf = torch.from_numpy(cmp_surf).permute((0, 3, 1, 2)).float().div(255)
    prj_cmp_path = myutils.join(save_path, os.path.split(pth_path)[1][:-4])
    if not os.path.exists(prj_cmp_path):
        os.makedirs(prj_cmp_path)
    if not os.path.exists(prj_cmp_path): os.makedirs(prj_cmp_path)
    st = time.time()
    for i in range(len(os.listdir(desire_test_path))):
        desire_test = readImgsMT(desire_test_path, index=[i], size=args.get('size')).to(device)
        cam_surf_test = cmp_surf.expand_as(desire_test).to(device)
        with torch.no_grad():
            compen_model.simplify(cam_surf_test[0, ...].unsqueeze(0))
            compen_model.eval()
            prj_cmp_test = compen_model(desire_test, cam_surf_test).detach()  # compensated prj input image x^{*}
            saveImg(prj_cmp_test, prj_cmp_path, i)  # compensated testing images, i.e., to be projected to the surface
        del desire_test, cam_surf_test
        nt = time.time()
        myutils.process("compensating image...", i, len(os.listdir(desire_test_path)), st, nt)
if __name__=="__main__":
    cmpImages(args['pth_path'], args['desire_img_path'], args["cmp_surf_path"], args["save_path"] )
