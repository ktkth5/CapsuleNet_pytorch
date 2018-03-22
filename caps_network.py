import torch
import torch.nn as nn
import torchvision.models as models

from primarycaps_layer import PrimaryCaps_Layer
from digitcaps_layer import DigitCaps_Layer
# from SENet import se_resnet152


class Caps_Net(nn.Module):

    def __init__(self, in_channels=2048, out_channels=128, capsule_n_=3*3,
                 capsule_size=5, out_capsule_n=55, out_capsule_size=1):
        super(Caps_Net, self).__init__()

        trained_model = models.resnet50()

        lists = list(trained_model.children())[:-2]
        self.model = nn.Sequential(*lists)
        for param in self.model.parameters():
            param.requires_grad = False

        self.primary_layer = PrimaryCaps_Layer(in_channels = in_channels,
                                               out_channels = out_channels,
                                               out_capsule_n = capsule_n_,
                                               out_capsule_size = capsule_size)
        self.digit_layer = DigitCaps_Layer(in_capsule_n = out_channels*capsule_n_,
                                           in_capsule_size = capsule_size,
                                           out_capsule_n = out_capsule_n,
                                           out_capsule_size = out_capsule_size)

    def forward(self, x):
        x = self.model(x)

        u = self.primary_layer(x)
        # print(u)
        u = self.digit_layer(u)
        # print(u)
        return u

if __name__=="__main__":
    model = Caps_Net()
    # print(model.primary_layer.parameters())
    # for param in model.digit_layer.parameters():
    #     print(param)
    # print(model.digit_layer.parameters())

    x = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
    out = model(x)
    print(out.shape)