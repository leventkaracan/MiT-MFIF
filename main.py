import os
import argparse

from torch.backends import cudnn
from model import FocusMapGenerator
from model import MultiscaleDiscriminator as DiscriminatorFusion
from data_loader import alpha_matte_AB
from torchvision import transforms as T
import torch.utils.data as data
import torch
from torch.nn import functional as F
import torch.nn as nn
from torchvision.utils import save_image
from loss_lib import GANLoss
import numpy as np
from core.spectral_norm import spectral_norm as _spectral_norm


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)


    # Data loader.
    train_loader = None
    test_loader = None

    print('^^^^^^ train ^^^^')
    #img_root = "/data/levent/mfif_dataset/" 
    img_root = config.root_traindata

    transform = []
    transform2 = []
    #crop_size = 178
    #image_size = 256
    crop_size = config.crop_size
    image_size = config.image_size

    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    
    transform2.append(T.CenterCrop(crop_size))
    transform2.append(T.Resize(image_size))
    transform2.append(T.ToTensor())
    transform2.append(T.Lambda(lambda x: x.repeat(3,1,1)))
    transform2 = T.Compose(transform2)
    
    train_data = alpha_matte_AB(img_root, transform, transform2)
    train_loader = data.DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)
    G = FocusMapGenerator().cuda()
    D = DiscriminatorFusion(input_nc=7, use_sigmoid=False, ndf=64, norm_layer=nn.BatchNorm2d).cuda()
    #g_lr = 0.00002
    #d_lr = 0.00002

    if config.resume:
      model_path_G = os.path.join(config.model_save_dir, config.resume_model, "_G_" + str(config.resume_epoch) + '_' + str(config.resume_iter))
      model_path_D = os.path.join(config.model_save_dir, config.resume_model, "_D_" + str(config.resume_epoch) + '_' + str(config.resume_iter))
      G.load_state_dict(torch.load(model_path_G))
      D.load_state_dict(torch.load(model_path_D))

    g_lr = config.g_lr
    d_lr = config.d_lr
    beta1 = config.beta1
    beta2 = config.beta2
    
    optimG = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(beta1, beta2))
    optimD = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(beta1, beta2))

    criterionGAN = GANLoss(use_lsgan=True, target_real_label=1.0)
    l1_loss = nn.L1Loss()
  
    total_iteration = 0
    update_lr_iteration = 0

    f_log = open(os.path.join(config.log_dir, "loss_log.txt"), 'w+')
    save_filename = config.model_save_dir + config.model_name
    num_epoch = config.num_epoch
    for epoch in range(num_epoch): 
        avg_gen_loss = 0.0
        avg_dis_fake_loss = 0.0
        avg_dis_real_loss = 0.0
        avg_grad_pen = 0.0
        avg_img_loss = 0.0
        avg_gan_feat_loss = 0.0
        it =  0
        itg = 0

        
        for img_A, img_B, foc_map, GT in train_loader:
            img_A = img_A.cuda()
            img_B = img_B.cuda() 
            foc_map = foc_map.cuda()
            GT = GT.cuda()

            img_mean = (img_A + img_B)*0.5;
            img_AB_mean = torch.cat((img_mean.unsqueeze(0), img_mean.unsqueeze(0)),dim=0)
            img_AB_mean = img_AB_mean.permute(1, 0, 2, 3, 4)

            GT2 = torch.cat((GT.unsqueeze(0),GT.unsqueeze(0)), dim=0)
            GT2 = GT2.permute(1,0,2,3,4)
            b, t, c, h, w = GT2.size()
            GT_mf_img = GT2.contiguous().view(b*t, c, h, w)
            pred_focmap = G(img_A, img_B)


            gen_loss = 0
            dis_loss = 0
            total_iteration +=1 

            # discriminator adversarial loss
            real_vid_feat = D(torch.cat((img_A, img_B, foc_map), 1))
            fake_vid_feat = D(torch.cat((img_A, img_B, pred_focmap.detach()),1))

            dis_real_loss = criterionGAN(real_vid_feat, True)
            dis_fake_loss = criterionGAN(fake_vid_feat, False)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2

        
            optimD.zero_grad()
            dis_loss.backward()
            optimD.step()
            avg_dis_fake_loss += dis_fake_loss.item()   
            avg_dis_real_loss += dis_real_loss.item()
            it+=1

            # generator adversarial loss
            if total_iteration%1 == 0:
                real_vid_feat = D(torch.cat((img_A, img_B, foc_map), 1))
                gen_vid_feat = D(torch.cat((img_A, img_B, pred_focmap), 1))
                gan_loss = criterionGAN(gen_vid_feat, True)
                gan_loss = gan_loss * 1
                gen_loss += gan_loss

                # generator l1 loss
                img_loss = l1_loss(pred_focmap, foc_map)
                gen_loss += 1 * img_loss
                optimG.zero_grad()
                gen_loss.backward()
                optimG.step()    
                avg_gen_loss+=gen_loss.item()  
                avg_img_loss+=img_loss.item()
                itg+=1


  
                
            if(it % 20 == 0):
                save_image((img_A + 1)*0.5, os.path.join(config.sample_dir, "%d_A.png"%epoch))
                save_image((img_B + 1)*0.5, os.path.join(config.sample_dir, "%d_B.png"%epoch))
                save_image((pred_focmap), os.path.join(config.sample_dir, "%d_pred.png"%epoch))
                save_image(foc_map, os.path.join(config.sample_dir, "%d_gt.png"%epoch))
                f_log.writelines('Epoch [%d/%d], Iter [%d/%d], gen_loss: %.4f, dis_real_loss: %.4f,  dis_fake_loss: %.4f, img_loss: %.4f\n'% (epoch, num_epoch, it, len(train_loader), avg_gen_loss/(itg), avg_dis_real_loss/it, avg_dis_fake_loss/it, avg_img_loss/(itg)));
                f_log.flush();
            if(it%100==0):
                torch.save(G.state_dict(), save_filename + "_G_" + str(epoch))
                torch.save(D.state_dict(), save_filename + "_D_" + str(epoch))
                torch.save(G.state_dict(), save_filename + "_G_" + str(epoch) + "_" + str(it))
                torch.save(D.state_dict(), save_filename + "_D_" + str(epoch) + "_" + str(it))
            if(total_iteration%1==0):
                print('Epoch [%d/%d], Iter [%d/%d], gen_loss: %.4f, dis_real_loss: %.4f,  dis_fake_loss: %.4f, img_loss: %.4f'% (epoch, num_epoch, it, len(train_loader), avg_gen_loss/(itg), avg_dis_real_loss/it, avg_dis_fake_loss/it, avg_img_loss/(itg)))


 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model name
    parser.add_argument('--model_name', type=str, default='mfif')

    # Model configuration.
    parser.add_argument('--crop_size', type=int, default=178, help='crop size for the alpha_matte_AB dataset')
    parser.add_argument('--image_size', type=int, default=256, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size') # Select the appropriate batch size according to the GPU memory
    parser.add_argument('--num_epoch', type=int, default=10, help='number of epochs')   
    parser.add_argument('--g_lr', type=float, default=0.00002, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.00002, help='learning rate for D')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_model', type=int, default=None, help='resume training from this model')
    parser.add_argument('--resume', type=bool, default=False, help='resume from a checkpoint' )
    parser.add_argument('--num_workers', type=int, default=0) 
    
    # Directories.
    parser.add_argument('--root_traindata', type=str, default='./dataset')
    parser.add_argument('--model_save_dir', type=str, default='./model')
    parser.add_argument('--sample_dir', type=str, default='./samples')
    parser.add_argument('--log_dir', type=str, default='./log')

    config = parser.parse_args()
    print(config)
    main(config)
    
