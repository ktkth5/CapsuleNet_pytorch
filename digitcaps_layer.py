import torch
import torch.nn as nn
from torch.autograd import Variable

# in_units : 5
# in_units_size : 128*3*3
# n_units : 55
# unit_size : 128

# input: (batch_size, 128, 5, 3, 3)
# output : (batch_size, 55, 128) - (batch_size, 55)

# 55 → 1とすれば最後の確率を求める式となる。

class DigitCaps_Layer(nn.Module):

    def __init__(self, in_capsule_n=128*3*3, in_capsule_size=5,
                 out_capsule_n=55, out_capsule_size=1):
        super(DigitCaps_Layer, self).__init__()

        self.in_capsule_n = in_capsule_n
        self.in_capsule_size = in_capsule_size
        self.out_capsule_n = out_capsule_n
        self.out_capsule_size = out_capsule_size
        self.W_ = nn.Parameter(torch.randn(1, in_capsule_n, 1))
        self.b_ij_ = nn.Parameter(torch.zeros((out_capsule_n, out_capsule_size)))


    def forward(self, u):
        """
        :param x: (batch_size, out_capsule_size, out_capsule)
                    (batch_size, 5, 128,3,3)
        :return: (batch_size, out_capsule)
        """
        batch_size = u.size(0)
        u_ = u.contiguous()
        u_ = u_.view(batch_size, self.in_capsule_size, self.in_capsule_n)
        # u_ = u.view(batch_size, self.in_capsule_size, self.in_capsule_n)
        W_ = torch.cat([self.W_]*batch_size, dim=0)
        u_hat_ = torch.matmul(u_, W_)
        # print(u_hat_.shape)
        u_out_ = self._routing(u_hat_)
        # print(u_out_.shape)
        return u_out_


    def _routing(self,u_hat):
        """
        :param x: (batch_size, 128, 5)
                    (batch_size, 64, 128)
        :return: (batch_size, 55,128)
        """
        batch_size = u_hat.size(0)
        softmax = nn.Softmax(dim=1)

        b_ij_ = self.b_ij_.expand((batch_size, self.out_capsule_n, self.in_capsule_size))
        # u_hat_ = torch.transpose(u_hat, 1, 2)

        iteration = 2
        for r in range(iteration):
            c_i_ = softmax(b_ij_)

            s_j_ = torch.matmul(c_i_, u_hat)

            v_j_ = self.squash(s_j_)

            b_ij_ = b_ij_ + torch.matmul(v_j_, u_hat.transpose(1,2))

        return v_j_

    def squash(self, s):
        """
        :param s: (batch_size, 55, 128)
        :return: (batch_size, 55)
        """
        norm_sq = torch.sum(s**2, dim=2, keepdim=True)
        # print(norm_sq.shape)
        norm = torch.sqrt(norm_sq)
        s = (norm/(1+norm_sq)) * s
        return s

if __name__=="__main__":
    c = DigitCaps_Layer()
    input = torch.autograd.Variable(torch.randn(1,128,5,3,3))
    for i in range(5):
        m = c(input)
    # x = c._routing(m)
    # print(x.shape)
    print(m.shape)