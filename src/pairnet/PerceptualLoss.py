import torch
import os
import torch.nn as nn
import torchvision.models as models
l2_fun = nn.MSELoss()
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3'
device_ids = [0, 1, 2]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model():
    """
    :brief :this method is used to get the check points in VGG,they respectively are 3rd and 5th max pooling layer's
            output
    :return checkpoint1,checkpoint2:checkpoint is the 3rd max pooling layer's output ==> [batch_size,3,64,64]
            checkpoint2 is the 5th max pooling layer's output ==> [batch_size,3,8,8]
    """
    checkpoint1 = models.vgg16(pretrained=True).features[:16]
    checkpoint2 = models.vgg16(pretrained=True).features[:30]
    checkpoint1, checkpoint2 = checkpoint1.eval(), checkpoint2.eval()
    checkpoint1.cuda()
    checkpoint2.cuda()
    return checkpoint1, checkpoint2

def extract_feature(checkpoint1, checkpoint2, x, y):
    """
    :param checkpoint1: the output tensor of vgg16 3th max pool layer
    :param checkpoint2: the output tensor of vgg16 5th max pool layer
    :param x: the prediction input tensor with size of [channels,width,height]==>[batch_size,3,256,256]
    :param y: the projector input tensor with size of [batch_size,channels,width,height]==>[batch_size,3,256,256]
    :return pred_predict,prj_predict:the prediction lists of pred and prj tensor,their size respectively are
            [[3th max pool][5th max pool]],[[3th max pool][5th max pool]]==>[[3,64,64][3,8,8]] and  [[3,64,64][3,8,8]]
    """
    checkpoint1.eval()
    checkpoint2.eval()
    feature1 = checkpoint1(x)
    feature2 = checkpoint2(x)
    feature1 = feature1.data.cpu().numpy()
    feature2 = feature2.data.cpu().numpy()
    pred_predict = [feature1, feature2]
    feature1 = checkpoint1(y)
    feature2 = checkpoint2(y)
    feature1 = feature1.data.cpu().numpy()
    feature2 = feature2.data.cpu().numpy()
    prj_predict = [feature1, feature2]
    return pred_predict, prj_predict

def computeLoss(pred_predict, prj_predict):
    """
    :brief : given the pred prediction tensor and the prj prediction tensor to compute the loss of perceptual
    :param pred_predict: the pred data feed to the VGG model and produce the prediction ==> [[batch_size,256,64,64][batch_size,512,16,16]]
    :param prj_predict: the projector data feed to the VGG model and produce the prediction ==> [[batch_size,256,64,64][batch_size_512,16,16]]
    :return l_perc:the perceptual loss of pred and prj data,which is a float
    """
    #3rd max pooling layer's prediction
    pred_checkpoint1 = torch.from_numpy(pred_predict[0]).to(device)
    prj_checkpoint1 = torch.from_numpy(prj_predict[0]).to(device)
    #5th max pooling layer's prediction
    pred_checkpoint2 = torch.from_numpy(pred_predict[1]).to(device)
    prj_checkpoint2 = torch.from_numpy(prj_predict[1]).to(device)
    # print(pred_checkpoint1.shape, prj_checkpoint1.shape)
    #3rd loss
    l2_loss_ckp1 = l2_fun(pred_checkpoint1, prj_checkpoint1)
    #5th loss
    l2_loss_ckp2 = l2_fun(pred_checkpoint2, prj_checkpoint2)
    l_perc = (l2_loss_ckp1+l2_loss_ckp2)/2.
    return l_perc

def perceptualLoss(prediction_batch, prj_train_batch):
    """
    :param prediction_batch: the output of compenet with shape of [batch_size,3,256,256]
    :param prj_train_batch: the projector pictures with shape of [batch_size,3,256,256]
    :return:
    """
    ckp1, ckp2 = get_model()
    pred, prj = extract_feature(ckp1,ckp2,prediction_batch,prj_train_batch)
    l_perc = computeLoss(pred,prj)
    return l_perc