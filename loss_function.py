import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Loss_Func(nn.Module):

    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_sym=0.5):
        super(Loss_Func, self).__init__()

        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_sym = lambda_sym

    def forward(self, result, targets, average=True):
        t = torch.zeros(result.size()).long()
        t = t.scatter_(1, targets.data.view(-1, 1), 1)
        if targets.is_cuda:
            t = t.cuda()
        targets = Variable(t)
        losses = targets.float() * F.relu(self.m_plus - result).pow(2) + \
                 self.lambda_sym * (1. - targets.float()) * F.relu(result - self.m_minus).pow(2)
        return losses.mean() if average else losses.sum()
