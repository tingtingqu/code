# -*- coding:utf-8 -*-
import os
import time
import cv2
from collections import OrderedDict
from PIL import Image
import numpy as np
from torch.autograd import Variable

np.seterr(divide='ignore', invalid='ignore')
# np.set_printoptions(threshold='nan')
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import shutil
from deeplab import Deeplab_v3_plus
from cal_iou import evaluate
from unet_model1 import UNet
from EffUNet import EffUNet
from MACUNet import MACUNet





#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# path = '/home/weikai/new_sdb/jk/rsi_data/val/images/'
# path = '/home/kawhi/Desktop/rsi_data/train/images_jpg/'

#t_start = time.clock()
t_start = time.perf_counter()
def img_transforms(img):
    # img, label = random_crop(img, label, crop_size)
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.315, 0.319, 0.470), std=(0.144, 0.151, 0.211))
    ])
    img = transform(img)
    return img

def find_new_file(dir):
    if os.path.exists(dir) is False:
    
        os.mkdir(dir)
        dir = dir

    file_lists = os.listdir(dir)
    file_lists.sort(key=lambda fn: os.path.getmtime(dir + fn)
    if not os.path.isdir(dir + fn) else 0)
    if len(file_lists) != 0:
        file = os.path.join(dir, file_lists[-1])
        return file
    else:
        return None

def label_mapping(label_im):
    #colorize = np.zeros([5, 3], dtype=np.int64)
    colorize = np.zeros([6, 3], dtype=np.int64)
    colorize[0, :] = [255, 255, 255]
    colorize[1, :] = [0, 0, 255]
    colorize[2, :] = [0, 255, 255]
    colorize[3, :] = [0, 255, 0]
    colorize[4, :] = [255, 255, 0]
    colorize[5, :] = [255 ,0, 0]



    label = colorize[label_im, :].reshape([label_im.shape[0], label_im.shape[1], 3])
    return label

def predict(net, im): # 预测结果
    # cm = np.array(colormap).astype('uint8')
    with torch.no_grad():
        #im = im.unsqueeze(0).cuda()
        # im = im.unsqueeze(0).cuda()
        im = im.unsqueeze(0).to(device)
        output = net(im).int()
        pred = output.max(1)[1].squeeze().cpu().data.numpy()
        pred_ = label_mapping(pred)
    return pred_, pred

def pred_image(p, img_path, model_dir, output, output_gray, path):

    img = Image.open(path+p)
    # lab = Image.open((path+p).replace('images', 'gt'))
    imsw, imsh = img.size

    crop_size = 800

    xw = int(imsw / crop_size)
    xh = int(imsh / crop_size)

    new_size = [(xw + 1) * crop_size, (xh + 1) * crop_size]
    new_img = img.resize((new_size[0], new_size[1]), Image.ANTIALIAS)
    # new_lab = lab.resize((new_size[0], new_size[1]), Image.ANTIALIAS)
    pred = np.zeros((new_size[1], new_size[0],3))
    all_gray = np.zeros((new_size[1], new_size[0]))

   
    #state_dict = torch.load(find_new_file(model_dir),map_location='cuda:0')
    state_dict = torch.load(find_new_file(model_dir), map_location='cpu')
  
    #state_dict = torch.load('./pth_attention_fuzzy/fcn-deconv-150.pth')

    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    
    num_class = 6
    #PSPNet
    #net = FuzzyNet(6,False)
    #net = UNet(2)
    #net = FCN8s(6)
    #net = SegNet(3, 6)
    #net = UNet(6,True)
    net = EffUNet()
    #net = MANet(3, 6)
    #net = MACUNet(3, 6)
    #net = MSResNet(3,6)
    #net = PSPNet(6,True)
    #net = Deeplab_v3_plus()
    #net = LinkNet()
    #net = Segnet_unet(3,6)
          
    net.load_state_dict(new_state_dict,False)
    #net.cuda()
    net.to(device)
    net.eval()
   
    for i in range(xh+1):
        for j in range(xw+1):
            # img2 = new_img.crop((crop_size * i, crop_size * j, crop_size * (i + 1), crop_size * (j + 1)))
            img2 = new_img.crop((crop_size * j, crop_size * i, crop_size * (j + 1), crop_size * (i + 1)))  #hengzhede
            # lab2 = new_lab.crop((crop_size * j, crop_size * i, crop_size * (j + 1), crop_size * (i + 1)))  #hengzhede
            name = img_path+str(i)+'_src_'+str(j)+'.png'

            img2.save(name)
            # lab2.save((name.replace('src','gt')).replace('jpg', 'png'))
            # batch_images = np.empty((1, 3, crop_size, crop_size))
            image = Image.open(name)
            # image = image.resize(input_hw[0:2], Image.NEAREST)
            # image_np = np.asarray(image, dtype=np.uint8)
            image_np = img_transforms(image)
            # batch_images = image_np
            # val_images = get_horse_generator(name, batch_size=1, input_hw=(224, 224, 3),
            #                                          mask_hw=(224, 224, 2))
            # image_np = np.asarray(img2, dtype=np.uint8)
            # image_np = norm(image_np)
            # batch_images = np.empty((1, crop_size, crop_size, 3))
            # batch_images[0] = image_np
            #res, gray = predict(net, Variable(torch.Tensor(image_np)).cuda())
            res, gray = predict(net, Variable(torch.Tensor(image_np)).to(device))
            im1 = Image.fromarray(np.uint8(res))
            im1.save((name.replace('src','pred')))
            im2 = Image.fromarray(np.uint8(gray))
            im2.save((name.replace('src', 'gray')))
            # print('%s, %s' %(i,j))
            # pred[crop_size * j:crop_size * (j + 1), crop_size * i:crop_size * (i + 1)] = pred_label
            pred[crop_size * i:crop_size * (i + 1), crop_size * j:crop_size * (j + 1)] = res
            all_gray[crop_size * i:crop_size * (i + 1), crop_size * j:crop_size * (j + 1)] = gray
    # cv2.imwrite((output+p[0:-4] + '.tif').replace(' (2)', '_label'), np.uint8(pred))

    result_img = Image.fromarray(np.uint8(pred))
    result_img = result_img.resize((imsw, imsh))
    result_img.save(output+p[0:-4] + '.tif')
    result_img_gray = Image.fromarray(np.uint8(all_gray))
    result_img_gray = result_img_gray.resize((imsw, imsh))
    result_img_gray.save(output_gray + p[0:-4] + '.tif')

def test_my():
    #path = './data/big_data/test/img/'
    #path = './1/img/'
    #path = './test fuzzy module/1/'
    #path = './data/big_data/test/img/'
    path = './testda/'
    #path = './data_big_village/'
    #path = './data_postdam1/img1/'
    #path = './data_postdam/JPEGImages/'

    #path = './data/JPEGImages/'
    #path = './data/big_data/test/img/'
    #model_dir = './pth_he/'
    #path = './testda/'
    #model_dir = './pth_attention_unet/'
    model_dir = './pth_1/b7/'

    #output_gray = './test_fuzzy_ISPRS_pth1_gray/'
    #output = './test_ISPRS_time/'
    #output_gray = './test_ISPRS_time_gray/'
    #output = './segnet_ISPRS/'
    #output_gray = './segnet_ISPRS_gray/'
    #output = './sune_fuzzy1_1000/'
    #output_gray = './sune_fuzzy1_1000_gray/'
    #output = './big/'
    output = './output_village/b7/5811/'
    #output = './attention2_1000/'
    #output_gray = './biggray'
    output_gray = './output_village/b7/5811gray600/'
    #output_gray = './attention2_1000gray/'
    if os.path.exists(output):
        shutil.rmtree(output)
        os.mkdir(output)
        print('delete sucessed')
    else:
        os.mkdir(output)
    if os.path.exists(output_gray):
        shutil.rmtree(output_gray)
        os.mkdir(output_gray)
        print('delete sucessed')
    else:
        os.mkdir(output_gray)
    f = os.listdir(path)
    for i in range(len(f)):
        img_path = output + f[i][0:-4] + '/'
        os.mkdir(img_path)
        print(f[i])
        pred_image(f[i], img_path, model_dir, output, output_gray, path)
    #gt = './data/big_data/test/gt/'
    #t_end = time.clock()
    t_end = time.perf_counter()
    run_time = t_end - t_start
    print(run_time)
    gt = './data/big_data/test/gt/'
    #gt = './data_postdam/SegmentationClassAug/'
    #gt = './testdagray/'
    #gt = './data/SegmentationClass/'
    iou, acc = evaluate(output_gray, gt, 6)
    return iou, acc

if __name__ == '__main__':

    iou, acc = test_my()





