import torch
import torch.nn as nn



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, nz, ngf):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.main = nn.Sequential(
            nn.ConvTranspose1d(self.nz, self.ngf * 8, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose1d(self.ngf * 8, self.ngf * 4, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(self.ngf * 4, self.ngf * 2, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose1d(self.ngf * 2, self.ngf * 1, bias=False),
            nn.BatchNorm1d(ngf * 1),
            nn.ReLU(True),
            nn.ConvTranspose1d(self.ngf * 1, self.nz, bias=False),
            nn.BatchNorm1d(self.nz * 1),
            nn.Tanh()
        )

    def forward(self, net):
        net = self.main(net)
        return net


class Infer(nn.Module):
    def __init__(self, nz, ngf):
        super(Infer, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.main = nn.Sequential(
            nn.ConvTranspose1d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose1d(self.ngf * 8, self.ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            nn.Tanh()
        )

    def forward(self, net):
        net = self.main(net)
        return net


if __name__=='__main__':
    pass