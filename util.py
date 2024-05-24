from torchvision import utils
import torch
import argparse
import yaml
from PIL import Image


def load_opt():
    with open("config.yml", mode='r') as f:
        option = yaml.load(f, Loader=yaml.FullLoader)
    return option


def parse_args(opt):
    parser = argparse.ArgumentParser(description="Args Parser")
    parser.add_argument('--bs', type=int, default=opt['train']['batch_size'], help="batch size")
    parser.add_argument('--lr', type=float, default=opt['train']['lr'], help="initial lr")
    parser.add_argument('--betas1', type=float, default=opt['train']['betas1'], help="betas1")
    parser.add_argument('--betas2', type=float, default=opt['train']['betas2'], help="betas2")
    parser.add_argument('--eps', type=float, default=opt['train']['eps'], help="eps")
    parser.add_argument('--weight_decay', type=float, default=opt['train']['weight_decay'], help="weight decay")
    parser.add_argument('--optim_step', type=int, default=opt['train']['optim_step'], help="optim step")
    parser.add_argument('--optim_gamma', type=float, default=opt['train']['optim_gamma'], help="optim gamma")

    parser.add_argument('--epoch_num', type=int, default=opt['train']['epoch_num'], help="epoch number")
    parser.add_argument('--save_freq', type=int, default=opt['train']['save_freq'], help="frequency to save")
    parser.add_argument('--device', type=str, default='cuda', help="device")
    parser.add_argument('--train_data_dir', type=str, default=opt['data']['data_dir'], help="train data dir")
    parser.add_argument('--load_dir', type=str, default='', help="checkpoint dir")
    parser.add_argument('--data_dim', type=int, default=1, help="dim num of data")
    parser.add_argument('--code_dim', type=int, default=opt['net']['code_size'], help="dim num of code")

    return parser.parse_args()


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_model(net, model_dir):
    net.load_state_dict(torch.load(model_dir)['net'])


def save_image_from_tensor(t, filename):
    while len(t.shape) < 4:
        t = t.unsqueeze(0)
    if t.shape[-1] == 1 or t.shape[-1] == 3:
        t = t.permute(0, 3, 1, 2)
    t = t.clone().detach()
    t = t.to(torch.device('cpu'))
    utils.save_image(t, filename)


opt = load_opt()
args = parse_args(opt)


if __name__ == '__main__':
    print(args)