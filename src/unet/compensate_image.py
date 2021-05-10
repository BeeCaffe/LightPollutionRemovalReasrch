from src.unet.my_trainutils import *
import src.unet.utils as utils
import time
__DEBUG=False
os.environ['CUDA_VISIBLE_DEVICES']='0'
device_ids = [0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()>=1:
    print('Valid with', torch.cuda.device_count(),'GPU...')
else:
    print('Valid with CPU...')
args = {
    "desire_img_path": r'I:\Backup\Images\projector/', #images path of which you suppose to compensate
    "save_path": 'output/unet/', #directory of save compensated images
    'checkpoint_path': 'checkpoint/unet/',
    "size": (768, 1024), #compensated image's size
}

pth_list = os.listdir(args['checkpoint_path'])

for pth in pth_list:
    pth_path = args['checkpoint_path'] + pth
    if not os.path.exists(pth_path):
        print("error : xxx.pth path not exist\n")
        continue
    model = torch.load(pth_path)
    torch.cuda.empty_cache()
    desire_test_path = args["desire_img_path"]
    assert os.path.isdir(desire_test_path), 'images and folder {:s} does not exist!'.format(desire_test_path)
    prj_cmp_path = utils.join(args['save_path'], os.path.split(pth_path)[1][:-4])
    if not os.path.exists(prj_cmp_path):
        os.makedirs(prj_cmp_path)
    if not os.path.exists(prj_cmp_path): os.makedirs(prj_cmp_path)
    st = time.time()
    for i in range(len(os.listdir(desire_test_path))):
        desire_test = readImgsMT(desire_test_path, index=[i], size=args.get('size')).to(device)
        with torch.no_grad():
            model.eval()
            compen = model(desire_test).detach()
            saveImg(compen, prj_cmp_path, i)
        del desire_test,
        nt = time.time()
        utils.process("compensating image...", i, len(os.listdir(desire_test_path)), st, nt)