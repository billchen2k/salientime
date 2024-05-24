import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, data_dim=1, code_dim=64):
        super(Encoder, self).__init__()
        self.data_dim = data_dim
        self.code_dim = code_dim // 8

        self.conv1 = nn.Conv2d(self.data_dim, 64, 4, 2, 1, bias=False)  # 256 - 128
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)  # 128 - 64
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)  # 64 - 32
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)  # 32 - 16
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, 4, 2, 1, bias=False)  # 16 - 8
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, 4, 2, 1, bias=False)  # 8 - 4
        self.bn6 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 512, 4, 2, 1, bias=False)  # 4 - 2
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, 4, 2, 1, bias=False)  # 2 - 1
        self.bn8 = nn.BatchNorm2d(512)

        self.code8 = nn.Conv2d(64, self.code_dim, kernel_size=128, stride=1, padding=0)
        self.code7 = nn.Conv2d(128, self.code_dim, kernel_size=64, stride=1, padding=0)
        self.code6 = nn.Conv2d(256, self.code_dim, kernel_size=32, stride=1, padding=0)
        self.code5 = nn.Conv2d(512, self.code_dim, kernel_size=16)
        self.code4 = nn.Conv2d(512, self.code_dim, kernel_size=8)
        self.code3 = nn.Conv2d(512, self.code_dim, kernel_size=4)
        self.code2 = nn.Conv2d(512, self.code_dim, kernel_size=2)
        self.code1 = nn.Conv2d(512, self.code_dim, kernel_size=1)

        self.act = nn.LeakyReLU()

    def forward(self, x):
        x1 = self.act(self.bn1(self.conv1(x)))
        x2 = self.act(self.bn2(self.conv2(x1)))
        x3 = self.act(self.bn3(self.conv3(x2)))
        x4 = self.act(self.bn4(self.conv4(x3)))
        x5 = self.act(self.bn5(self.conv5(x4)))
        x6 = self.act(self.bn6(self.conv6(x5)))
        x7 = self.act(self.bn7(self.conv7(x6)))
        x8 = self.act(self.bn8(self.conv8(x7)))

        code1 = self.code1(x8)
        code2 = self.code2(x7)
        code3 = self.code3(x6)
        code4 = self.code4(x5)
        code5 = self.code5(x4)
        code6 = self.code6(x3)
        code7 = self.code7(x2)
        code8 = self.code8(x1)
        code = torch.cat((code1, code2, code3, code4, code5, code6, code7, code8), dim=1)

        return code

class Decoder(nn.Module):
    def __init__(self, data_dim=1, code_dim=64):
        super(Decoder, self).__init__()
        self.data_dim = data_dim
        self.code_dim = code_dim // 8

        self.deconv1 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)  # 1 - 2
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)  # 2 - 4
        self.bn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)  # 4 - 8
        self.bn3 = nn.BatchNorm2d(512)
        self.deconv4 = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)  # 8 - 16
        self.bn4 = nn.BatchNorm2d(512)
        self.deconv5 = nn.ConvTranspose2d(1024, 256, 4, 2, 1, bias=False)  # 16 - 32
        self.bn5 = nn.BatchNorm2d(256)
        self.deconv6 = nn.ConvTranspose2d(512, 128, 4, 2, 1, bias=False)  # 32 - 64
        self.bn6 = nn.BatchNorm2d(128)
        self.deconv7 = nn.ConvTranspose2d(256, 64, 4, 2, 1, bias=False)  # 64 - 128
        self.bn7 = nn.BatchNorm2d(64)
        self.deconv8 = nn.ConvTranspose2d(128, self.data_dim, 4, 2, 1, bias=False)  # 128 - 256
        self.bn8 = nn.BatchNorm2d(self.data_dim)

        self.code1 = nn.ConvTranspose2d(self.code_dim, 512, kernel_size=1)
        self.code2 = nn.ConvTranspose2d(self.code_dim, 512, kernel_size=2)
        self.code3 = nn.ConvTranspose2d(self.code_dim, 512, kernel_size=4)
        self.code4 = nn.ConvTranspose2d(self.code_dim, 512, kernel_size=8)
        self.code5 = nn.ConvTranspose2d(self.code_dim, 512, kernel_size=16)
        self.code6 = nn.ConvTranspose2d(self.code_dim, 256, kernel_size=32, stride=1, padding=0)
        self.code7 = nn.ConvTranspose2d(self.code_dim, 128, kernel_size=64, stride=1, padding=0)
        self.code8 = nn.ConvTranspose2d(self.code_dim, 64, kernel_size=128, stride=1, padding=0)

        self.out_mask = nn.ConvTranspose2d(128, self.data_dim, 4, 2, 1, bias=False)  # 128 - 256
        self.bn_mask = nn.BatchNorm2d(self.data_dim)
        self.out_mask_act = nn.Sigmoid()

        self.act = nn.LeakyReLU()
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        code1, code2, code3, code4, code5, code6, code7, code8 = [x[:, i * self.code_dim: (i + 1) * self.code_dim, ...] for i in range(8)]
        d1 = self.act(self.bn1(self.deconv1(self.code1(code1))))
        d2 = self.act(self.bn2(self.deconv2(torch.cat((d1, self.code2(code2)), dim=1))))
        d3 = self.act(self.bn3(self.deconv3(torch.cat((d2, self.code3(code3)), dim=1))))
        d4 = self.act(self.bn4(self.deconv4(torch.cat((d3, self.code4(code4)), dim=1))))
        d5 = self.act(self.bn5(self.deconv5(torch.cat((d4, self.code5(code5)), dim=1))))
        d6 = self.act(self.bn6(self.deconv6(torch.cat((d5, self.code6(code6)), dim=1))))
        d7 = self.act(self.bn7(self.deconv7(torch.cat((d6, self.code7(code7)), dim=1))))
        d8 = self.out_act(self.bn8(self.deconv8(torch.cat((d7, self.code8(code8)), dim=1))))

        # d8_mask = self.out_mask_act(self.bn_mask(self.out_mask(torch.cat((d7, self.code8(code8)), dim=1))))
        return d8

class Discriminator(nn.Module):
    def __init__(self, data_dim=1):
        super(Discriminator, self).__init__()
        self.data_dim = data_dim

        self.net = nn.Sequential(
            nn.Conv2d(self.data_dim, 64, 4, 2, 1, bias=False),  # 256 - 128
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 128 - 64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 64 - 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 32 - 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1, 4, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.net(x)
        return x

class SalienTimeNet(nn.Module):
    def __init__(self, data_dim=1, code_dim=64):
        super(SalienTimeNet, self).__init__()
        self.data_dim = data_dim
        self.code_dim = code_dim

        self.encoder = Encoder(self.data_dim ,self.code_dim)
        self.decoder = Decoder(self.data_dim, self.code_dim)
        self.discriminator = Discriminator(self.data_dim)
        self.criterion_dis = nn.MSELoss(size_average=True)
        self.criterion_recons = nn.MSELoss(size_average=True)

    def __repr__(self):
        return f'data_dim: {self.data_dim}, code_dim: {self.code_dim}'

    def discriminate(self, x):
        x = self.discriminator(x)
        return x

    def forward(self, x):
        code = self.encoder(x)
        recons_x = self.decoder(code)
        dis_result_real = self.discriminate(x)
        dis_result_fake = self.discriminate(recons_x.detach())

        return x, code, recons_x, dis_result_real, dis_result_fake

    def forward_codeonly(self, x):
        return self.encoder(x)

    def calc_loss_dis(self, x, code, recons_x, dis_result_real, dis_result_fake):
        label_real = torch.ones_like(dis_result_real).to(x.device)
        label_real.requires_grad = False
        label_fake = torch.zeros_like(dis_result_fake).to(x.device)
        label_fake.requires_grad = False

        dis_real_loss = self.criterion_dis(dis_result_real, label_real)
        dis_fake_loss = self.criterion_dis(dis_result_fake, label_fake)
        dis_loss = dis_real_loss + dis_fake_loss

        return dis_loss

    def calc_loss_recons(self, x, code, recons_x, dis_result_real, dis_result_fake):
        label_real = torch.ones_like(dis_result_real).to(x.device)
        label_real.requires_grad = False

        recons_loss = self.criterion_recons(x, recons_x)
        recons_loss_dis = self.criterion_recons(self.discriminate(x), label_real)

        return recons_loss, recons_loss_dis
