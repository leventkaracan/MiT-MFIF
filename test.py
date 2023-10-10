import os
import argparse
from torch.backends import cudnn
from model import FocusMapGenerator
from data_loader import *
from torchvision import transforms as T
import torch.utils.data as data
import torch
from torch.nn import functional as F
from core.loss import AdversarialLoss
import torch.nn as nn
from torchvision.utils import save_image
from post_process import post_remove_small_objects
from post_process import threshold
from timeit import default_timer as timer
import time
import cv2
import skimage.io
import numpy as np


        
def main(config):
    output_root = os.path.join(config.root_result, config.test_dataset)
    if not os.path.exists(output_root):
      os.makedirs(output_root)
    transform = []

    #transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    print(config.root_testdata)
    img_root = os.path.join(config.root_testdata, config.test_dataset)
    if(config.test_dataset == "LytroDataset"):
      test_data = LytroDataset(img_root, transform)
    elif(config.test_dataset == "MFFW2"):
      test_data = MFFW2(img_root, transform)
        
    test_loader = data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=config.num_workers)
    G = FocusMapGenerator().cuda()

    model_path = os.path.join(config.root_model,config.model_name)

    G.load_state_dict(torch.load(model_path))
    G.eval()

    no = 1
    torch.backends.cudnn.enabled = True # solves cudnn error when testing
    
    for img_A, img_B in test_loader:
        with torch.no_grad():
            print(no)

            fusion_start_time = time.time()
            img_A = img_A.cuda()
            img_B = img_B.cuda()

            focus_map = G(img_A, img_B)

            fusion_used_time = time.time() - fusion_start_time
            print(fusion_used_time)
            save_image(focus_map, os.path.join(output_root, "%d_pred.jpg"%no))
            #our method does not need a post-processing
            #focus_map = threshold(focus_map, t=0.5).float()
            #focus_map = post_remove_small_objects(focus_map).float()


            fused_img = img_A * focus_map.cuda() + img_B * (1 - focus_map.cuda())
            save_image((img_A + 1)*0.5, os.path.join(output_root, "%d_A.png"%no))
            save_image((img_B + 1)*0.5, os.path.join(output_root,"%d_B.png"%no))
            save_image((focus_map), os.path.join(output_root,"%d_post.png"%no))
            fused_img = ((fused_img + 1)*0.5)
            save_image(fused_img, os.path.join(output_root,"%d_fused.jpg"%no))
            no+=1
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Test configuration.
    #mitmfif_relu_l11_relulayernorm_se16
    parser.add_argument('--model_name', type=str, default='mit-mfif_best')
    parser.add_argument('--test_dataset', type=str, default='LytroDataset')
    parser.add_argument('--num_workers', type=int, default=0) # win
    parser.add_argument('--root_testdata', type=str, default='./dataset/lytro/')
    parser.add_argument('--root_model', type=str, default='./model/')
    parser.add_argument('--root_result', type=str, default='./results/')
   
    config = parser.parse_args()
    print(config)
    main(config)
    
