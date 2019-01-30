import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, nz, nf):
        super(Generator, self).__init__()
        self.nz = nz
        self.nf = nf
        self.main = nn.Sequential(
            nn.Linear(self.nz, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(True),
            nn.Linear(200, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(True),
            nn.Linear(400, self.nf),
        )

    def forward(self, x):
        out = self.main(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, nf):
        super(Discriminator, self).__init__()
        self.ndf = nf
        self.main = nn.Sequential(
            nn.Linear(self.nf, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(0.2),
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 1),
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