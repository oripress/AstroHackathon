import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, nz, nc, ngf):
        super(Generator, self).__init__()
        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        self.main = nn.Sequential(
            nn.ConvTranspose1d(self.nz, self.ngf * 16, 8, 2, bias=False),
            nn.BatchNorm1d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose1d(self.ngf * 16, self.ngf * 8, 8, 4, 1, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose1d(self.ngf * 8, self.ngf * 4, 8, 4, 1, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(self.ngf * 4, self.ngf * 2, 8, 4, 1, bias=False),
            nn.BatchNorm1d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose1d(self.ngf * 2, self.ngf, 8, 4, 1, bias=False),
            nn.BatchNorm1d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose1d(self.ngf, self.nc, 8, 4, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.main(x)
        clip_size = (out.size(-1) - 8295) // 2
        return out[:, :, clip_size + 1:-clip_size]


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            nn.Conv1d(self.nc, self.ndf, 8, 4, bias=False),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2),
            nn.Conv1d(self.ndf, self.ndf * 2, 8, 4, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(self.ndf * 2, self.ndf * 4, 8, 4, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(self.ndf * 4, self.ndf * 8, 8, 4, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2),
            nn.Conv1d(self.ndf * 8, self.ndf * 16, 8, 4, bias=False),
            nn.BatchNorm1d(ndf * 16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(self.ndf * 16, self.nc, 8, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, net):
        net = self.main(net)
        return net


class Infer(nn.Module):
    def __init__(self, nc, ndf):
        super(Infer, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            nn.Conv1d(self.nc, self.ndf, 4, bias=False),
            nn.BatchNorm1d(ndf),
            nn.LeakyReLU(0.2),
            nn.Conv1d(self.ndf, self.ndf * 2, 4, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(self.ndf * 2, self.ndf * 4, 4, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(self.ndf * 4, self.ndf * 8, 4, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2),
            nn.Conv1d(self.ndf * 8, self.nc, 4, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.main(x).view(-1, 1).squeeze(1)
        return out


if __name__=='__main__':
    pass