import torch
import torch.nn as nn

# in_channel : 2048
# hidden_channel : 512
# out_channel : 128
# n_units : 5


## To make vecor u
class ConvUnit(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, bias=True):
        super(ConvUnit, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias
        )

    def forward(self, x):
        return self.conv(x)

class PrimaryCaps_Layer(nn.Module):

    def __init__(self, in_channels=2048, out_channels=128,
                 out_capsule_n=3*3, out_capsule_size=5,
                 kernel_size=5, stride=1):
        super(PrimaryCaps_Layer, self).__init__()

        # self.conv0 = nn.Conv2d(
        #     in_channels=in_channels,
        #     out_channels=512,
        #     kernel_size=1,
        #     stride=1,
        #     bias=True
        # )

        self.out_capsule_n = out_capsule_n
        self.out_capsule_size = out_capsule_size
        self.conv_layers = nn.ModuleList(
            [ConvUnit(in_channels, out_channels,
                      kernel_size, stride) for _ in range(out_capsule_size)])

    def forward(self, x):
        """
        :param x: (batch size, 2048, 7, 7)
        :return: (batch_size, 5, 128, 3, 3)
                 (batch_size, out_channel, n_units, unit_size)
        """
        # x = self.conv0(x)
        u = [self.conv_layers[i](x) for i in range(self.out_capsule_size)]
        # print(u[0].shape)
        u = torch.stack(u,dim=1)
        # print(u.shape)
        return u

if __name__=="__main__":
    c = PrimaryCaps_Layer()
    x = torch.autograd.Variable(torch.randn(2, 2048, 7, 7))
    #print(x)
    u=c(x)
    print(u.shape) # (batch_size,128,5,3,3)