from src.unet.my_trainutils import *
import src.unet.utils as utils
import torch
import time
DEBUG = True
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
    "desire_img_path": 'res/input/',#images path of which you suppose to compensate
    "save_path": 'output/attention_featuremap/',#directory of save compensated images
    "mask_pth": "checkpoint/pair_net/pair_net_l1+l2+ssim_4000_16_100_0_1.8_1.2_False-mask.pth",
    "size": (1088, 1920),#compensated image's size
}
desire_test_path=args.get('desire_img_path')
save_path=args.get('save_path')
mask_pth = args["mask_pth"]

if not os.path.exists(mask_pth):
    print("error : xxx.pth path not exist\n")
mask_net = torch.load(mask_pth)
torch.cuda.empty_cache()
desire_test_path = desire_test_path
assert os.path.isdir(desire_test_path), 'images and folder {:s} does not exist!'.format(desire_test_path)
save_path = utils.join(save_path, os.path.split(mask_pth)[1][:-4])
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_path): os.makedirs(save_path)
st = time.time()
for i in range(len(os.listdir(desire_test_path))):
    desire_test = readImgsMT(desire_test_path, index=[i], size=args.get('size')).to(device)
    with torch.no_grad():
        mask = mask_net(desire_test)
        saveImg(torch.clamp(mask, max=1), save_path, i)  # compensated testing images, i.e., to be projected to the surface
    del desire_test,
    nt = time.time()
    utils.process("compensating image...", i, len(os.listdir(desire_test_path)), st, nt)
