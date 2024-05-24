import torch
import torch.nn as nn
from model.stn import SalienTimeNet
from dataloader import SalienTimeDataset
from torch.utils.data import DataLoader
from util import args, save_image_from_tensor, load_model
import torch.optim as optim
import time
from torchvision import transforms
from PIL import Image
from skimage import io
import numpy as np
import tifffile


if __name__ == '__main__':
    device = args.device
    net = SalienTimeNet(data_dim=args.data_dim, code_dim=args.code_dim).to(device)
    print(net)
    net.train()
    
    
    params_trainable_ed = (list(filter(lambda p: p.requires_grad, list(net.encoder.parameters()) + list(net.decoder.parameters()))))
    params_trainable_dis = (list(filter(lambda p: p.requires_grad, list(net.discriminator.parameters()))))
    optimizer_ed = optim.Adam(params_trainable_ed,
                           args.lr,
                           betas=(args.betas1, args.betas2),
                           eps=args.eps,
                           weight_decay=args.weight_decay)
    optimizer_dis = optim.Adam(params_trainable_dis,
                           args.lr,
                           betas=(args.betas1, args.betas2),
                           eps=args.eps,
                           weight_decay=args.weight_decay)
    lr_scheduler_ed = optim.lr_scheduler.StepLR(optimizer_ed, args.optim_step, gamma=args.optim_gamma)
    lr_scheduler_dis = optim.lr_scheduler.StepLR(optimizer_dis, args.optim_step, gamma=args.optim_gamma)

    data_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip(),
                                         transforms.RandomCrop([256, 256]),
                                         transforms.ToTensor()])
    dataset = SalienTimeDataset(data_dir=args.train_data_dir, 
                                data_transform=data_transform)
    print(dataset)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=16, drop_last=True)

    for epoch_idx in range(1, args.epoch_num + 1):
        net.train()
        epoch_s_time = time.time()
        epoch_loss = [0, 0, 0]

        for data_idx, data in enumerate(dataloader):
            img = data['image'].to(device)[:, :args.data_dim]
            net_output = net(img)

            dis_loss = net.calc_loss_dis(*net_output)
            optimizer_dis.zero_grad()
            dis_loss.backward()
            optimizer_dis.step()

            recons_loss, recons_loss_dis = net.calc_loss_recons(*net_output)
            optimizer_ed.zero_grad()
            recons_loss_total = recons_loss + recons_loss_dis
            recons_loss_total.backward()
            optimizer_ed.step()

            epoch_loss[0] += dis_loss.item()
            epoch_loss[1] += recons_loss.item()
            epoch_loss[2] += recons_loss_dis.item()

            if device == 'cuda':
                img = img.cpu()
                torch.cuda.empty_cache()


        lr_scheduler_ed.step()
        lr_scheduler_dis.step()
        epoch_t_time = time.time()
        
        if epoch_idx % args.save_freq == 0:
            torch.save({'net': net.state_dict()}, f'./checkpoints/epoch{epoch_idx}_recons_loss_{epoch_loss[1] / len(dataset): .6f}')
            

        print(f'epoch {epoch_idx} -- ' \
              f'lr: {optimizer_ed.state_dict()["param_groups"][0]["lr"]: .12f}  ' \
              f'dis_loss: {epoch_loss[0] / len(dataset): .6f}  ' \
              f'recons_loss: {epoch_loss[1] / len(dataset): .6f}  ' \
              f'recons_loss_dis: {epoch_loss[2] / len(dataset): .6f}  ' \
              f'time: {epoch_t_time - epoch_s_time: .1f}')