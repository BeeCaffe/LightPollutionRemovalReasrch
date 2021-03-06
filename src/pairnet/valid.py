from src.unet.my_trainutils import *
import src.unet.utils as utils
import time
import src.unet.my_unet_model as unet_model
from time import localtime,strftime
__DEBUG=False
os.environ['CUDA_VISIBLE_DEVICES']='0'
device_ids = [0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()>=1:
    print('Valid with', torch.cuda.device_count(),'GPU...')
else:
    print('Valid with CPU...')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = {
    "desire_img_path": 'data/dataset_512_global_light/cmp/desire',#images path of which you suppose to compensate
    "save_path": 'data/dataset_512_global_light/cmp/res',#directory of save compensated images
    'pth_path': 'checkpoint/SAUnet-Global_l1+l2+ssim_3700_4_10_0.01_0.2_5000_0.0001.pth',
    "size": (1088, 1920),#compensated image's size
}

def cmpImages(pth_path,mask_pth,desire_test_path,save_path):
    if not os.path.exists(pth_path):
        print("error : xxx.pth path not exist\n")
    model = torch.load(pth_path)
    mask_net = torch.load(mask_pth)
    torch.cuda.empty_cache()
    desire_test_path = desire_test_path
    assert os.path.isdir(desire_test_path), 'images and folder {:s} does not exist!'.format(desire_test_path)
    prj_cmp_path = utils.join(save_path, os.path.split(pth_path)[1][:-4])
    if not os.path.exists(prj_cmp_path):
        os.makedirs(prj_cmp_path)
    if not os.path.exists(prj_cmp_path): os.makedirs(prj_cmp_path)
    st = time.time()
    for i in range(len(os.listdir(desire_test_path))):
        desire_test = readImgsMT(desire_test_path, index=[i], size=args.get('size')).to(device)
        with torch.no_grad():
            model.eval()
            mask = mask_net(desire_test)
            compen = model(desire_test, desire_test, desire_test, mask).detach()  # compensated prj input image x^{*}
            saveImg(compen, prj_cmp_path, i)  # compensated testing images, i.e., to be projected to the surface
        del desire_test,
        nt = time.time()
        utils.process("compensating image...", i, len(os.listdir(desire_test_path)), st, nt)
if __name__=="__main__":
    cmpImages(pth_path=args.get('pth_path'), desire_test_path=args.get('desire_img_path'), save_path=args.get('save_path'))