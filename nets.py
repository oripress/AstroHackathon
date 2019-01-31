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
            nn.Tanh()
        )

    def forward(self, x):
        out = self.main(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, nf):
        super(Discriminator, self).__init__()
        self.nf = nf
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
        return net.view(-1)


if __name__=='__main__':
    pass