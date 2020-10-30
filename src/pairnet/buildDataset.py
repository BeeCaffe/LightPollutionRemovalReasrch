import os
import cv2
import numpy as np
import time
import utils
args = {
    'cols': 512,
    'rows': 512,
    'max_nums': 5000,
    ###you need to change by yourself
    'camcap_path': 'H:\experiment data\\2019.12.13\camera-captured',#camera capture images path
    'grdtuth_path': 'H:\experiment data\\2019.12.13\ground-truth',#ground truth images path
    'dataset_root': 'data/dataset_512_global_light',#dataset root directory

    'cam_ref_path': 'cam/ref',#plain color images tilde(s)
    'cam_test_path': 'cam/test',#validation images \tilde(y)
    'cam_train_path': 'cam/train',#training images, \tilde(x)
    'prj_test_path': 'prj/test',#projector input validation images, i.e., y
    'prj_train_path': 'prj/train',#projector input training images, i.e., x
    'cmp_ref_path': 'cmp/surf',
    'train_nums': 3700,
    'test_nums': 100,
    'ref_grayscale': 128,
    'renamed': True,
    'gamma': 0,
}

def join(path1,path2):
    if not path1.endswith('/'):
        path1 += '/'
    return path1+path2

def buildDataset(camcap_path,grdtuth_path,dataset_root):

    cam_test_txt=join(args.get('dataset_root'), args.get('cam_test_path'))+'/../cam_test.txt'
    cam_train_txt=join(args.get('dataset_root'), args.get('cam_train_path'))+'/../cam_train.txt'
    prj_test_txt=join(args.get('dataset_root'), args.get('prj_test_path'))+'/../cam_test.txt'
    prj_train_txt=join(args.get('dataset_root'), args.get('prj_train_path'))+'/../cam_train.txt'

    if not os.path.exists(camcap_path):
        raise RuntimeError("camcap_path not exists")
    if not os.path.exists(grdtuth_path):
        raise RuntimeError("grdtuth_path not exists")
    if not os.path.exists(dataset_root):
        os.mkdir(dataset_root)

    if args.get('renamed') == False:
        utils.rename(camcap_path)
        utils.rename(grdtuth_path)

    cam_train_dir = join(args.get('dataset_root'), args.get('cam_train_path'))
    prj_train_dir = join(args.get('dataset_root'), args.get('prj_train_path'))
    cam_test_dir = join(args.get('dataset_root'), args.get('cam_test_path'))
    prj_test_dir = join(args.get('dataset_root'), args.get('prj_test_path'))

    if not os.path.exists(cam_train_dir):
        os.makedirs(cam_train_dir)
    if not os.path.exists(prj_train_dir):
        os.makedirs(prj_train_dir)
    if not os.path.exists(cam_test_dir):
        os.makedirs(cam_test_dir)
    if not os.path.exists(prj_test_dir):
        os.makedirs(prj_test_dir)

    #get all images name
    camcap_namelists=os.listdir(camcap_path)
    camcap_namelists.sort(key=lambda x: int(x[:-4]))
    grdtuth_namelists=os.listdir(grdtuth_path)
    grdtuth_namelists.sort(key=lambda x: int(x[:-4]))

    i = 0
    total=len(camcap_namelists)
    st = time.time()
    max_nums = args.get('max_nums')
    if max_nums>len(camcap_namelists) or max_nums>len(grdtuth_namelists):
        max_nums=min(len(camcap_namelists),len(grdtuth_namelists))

    train_nums=args.get('train_nums')
    test_nums=args.get('test_nums')
    if train_nums == 0 and test_nums == 0:
        train_nums=total*0.8
        test_nums=total*0.2

    #open txt file for writing
    cam_test_f=open(cam_test_txt, "w")
    cam_train_f=open(cam_train_txt, "w")
    prj_test_f=open(prj_test_txt, "w")
    prj_train_f=open(prj_train_txt, "w")

    cam_ref_dir = join(args.get('dataset_root'), args.get('cam_ref_path'))
    cmp_ref_path = join(args['dataset_root'], args['cmp_ref_path'])
    if not os.path.exists(cam_ref_dir):
        os.makedirs(cam_ref_dir)
    if not os.path.exists(cmp_ref_path):
        os.makedirs(cmp_ref_path)
    #camera reference image

    for cam_name, prj_name in zip(camcap_namelists, grdtuth_namelists):
        if i>=max_nums:
            break
        if args['gamma'] != 0:
            cam_img = cv2.imread(join(camcap_path, cam_name), cv2.IMREAD_UNCHANGED)
            prj_img = cv2.imread(join(grdtuth_path, prj_name), cv2.IMREAD_UNCHANGED)
            cam_img = gm.GammaCorrectByTable(cv2.resize(cam_img, (args.get('cols'), args.get('rows'))), args['gamma'])
            prj_img = cv2.resize(prj_img, (args.get('cols'), args.get('rows')))
        else:
            cam_img = cv2.imread(join(camcap_path, cam_name), cv2.IMREAD_UNCHANGED)
            prj_img = cv2.imread(join(grdtuth_path, prj_name), cv2.IMREAD_UNCHANGED)
            cam_img = cv2.resize(cam_img, (args.get('cols'), args.get('rows')))
            prj_img = cv2.resize(prj_img, (args.get('cols'), args.get('rows')))

        if i < train_nums:
            cv2.imwrite(join(prj_train_dir, str(i)+'.jpg'), prj_img)
            cv2.imwrite(join(cam_train_dir, str(i)+'.jpg'), cam_img)
            prj_train_f.write(join(prj_train_dir, str(i)+'.jpg')+'\n')
            cam_train_f.write(join(cam_train_dir, str(i)+'.jpg')+'\n')
        elif i>= train_nums and i<train_nums+test_nums:
            cv2.imwrite(join(cam_test_dir, str(i)+'.jpg'), cam_img)
            cv2.imwrite(join(prj_test_dir, str(i)+'.jpg'), prj_img)
            prj_test_f.write(join(prj_test_dir, str(i)+'.jpg')+'\n')
            cam_test_f.write(join(cam_test_dir, str(i)+'.jpg')+'\n')
        else: break


        nt=time.time()
        utils.process("making datset :", i, train_nums+test_nums, st, nt)
        i+=1
    cam_test_f.close()
    cam_train_f.close()
    prj_test_f.close()
    prj_train_f.close()
    print("buildDataset: DONE!")

if __name__=='__main__':
    buildDataset(args.get('camcap_path'),args.get('grdtuth_path'),args.get('dataset_root'))
