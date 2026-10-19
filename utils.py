import torch
import torch.nn as nn
import numpy as np

#这个 compute_rmse 函数的作用是计算 均方根误差 (RMSE)，它用于衡量两个数据（通常是预测值与真实值之间）的差异。
# 这个函数的输入是两个三维数组（通常表示图像或多维数据），输出是每个通道的 RMSE 和整体 RMSE。
def compute_rmse(x_true, x_pre):
    w, h, c = x_true.shape
    class_rmse = [0] * c# # 初始化一个列表，存储每个通道的 RMSE
    for i in range(c):# # 遍历每个通道，计算该通道的 RMSE
        class_rmse[i] = np.sqrt(((x_true[:, :, i] - x_pre[:, :, i]-1e-6) ** 2).sum() / (w * h))
    mean_rmse = np.array(class_rmse).mean()
    return class_rmse, mean_rmse

#这个函数用于计算 重构误差 (Reconstruction Error, RE)，通常用于评估图像（或其他多维数据）中预测值与真实值之间的差异。
# 它计算的是一个整体的均方根误差（RMSE），而不是按通道分开计算。
def compute_re(x_true, x_pred):
    img_w, img_h, img_c = x_true.shape
    return np.sqrt(((x_true - x_pred) ** 2).sum() / (img_w * img_h * img_c))

#这个函数用于计算 角度余弦差 (SAD, Spectral Angle Distance)，它是高光谱图像处理中常用的度量方式，用于衡量两个信号或光谱之间的角度差异。
# 具体而言，SAD 是通过计算两个向量之间的角度余弦来衡量它们的相似度，角度越小，两个向量的相似度越高。
def compute_sad(inp, target):
    p = inp.shape[-1]
    sad_err = [0] * p
    for i in range(p):
        inp_norm = np.linalg.norm(inp[:, i], 2)
        tar_norm = np.linalg.norm(target[:, i], 2)
        summation = np.matmul(inp[:, i].T, target[:, i])
        sad_err[i] = np.arccos(summation / (inp_norm * tar_norm)+1e-6)
    mean_sad = np.mean(sad_err)
    #sad_err: 一个列表，包含每个通道的 SAD 值。
    #mean_sad: 所有通道的平均 SAD 值。
    return sad_err, mean_sad


def Nuclear_norm(inputs):
    band, h, w = inputs.shape
    inp = torch.reshape(inputs, (band, h * w))
    out = torch.norm(inp, p='nuc')
    return out


class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()

    def __call__(self, inp, decay):
        inp = torch.sum(inp, 0, keepdim=True)
        loss = Nuclear_norm(inp)
        return decay * loss


class SumToOneLoss(nn.Module):
    def __init__(self, device):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float, device=device))
        self.loss = nn.L1Loss(reduction='sum')

    def get_target_tensor(self, inp):
        target_tensor = self.one
        return target_tensor.expand_as(inp)

    def __call__(self, inp, gamma_reg):
        inp = torch.sum(inp, 1)
        target_tensor = self.get_target_tensor(inp)
        loss = self.loss(inp, target_tensor)
        return gamma_reg * loss


class SAD(nn.Module):
    def __init__(self, num_bands):
        super(SAD, self).__init__()
        self.num_bands = num_bands

    def forward(self, inp, target):
        try:
            input_norm = torch.sqrt(torch.bmm(inp.view(-1, 1, self.num_bands),
                                              inp.view(-1, self.num_bands, 1)))
            target_norm = torch.sqrt(torch.bmm(target.view(-1, 1, self.num_bands),
                                               target.view(-1, self.num_bands, 1)))

            summation = torch.bmm(inp.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1))
            angle = torch.acos(summation / (input_norm * target_norm))

        except ValueError:
            return 0.0

        return angle

import numpy as np

class MinVolumn(nn.Module):
    def __init__(self, band, num_classes, delta):
        super(MinVolumn, self).__init__()
        self.band = band
        self.delta = delta
        self.num_classes = num_classes

    def __call__(self, edm):
        edm_result = torch.reshape(edm, (self.band, self.num_classes))
        edm_mean = edm_result.mean(dim=1, keepdim=True)
        loss = self.delta * ((edm_result - edm_mean) ** 2).sum() / self.band / self.num_classes
        return loss
class SID(nn.Module):
    def __init__(self, epsilon: float = 1e5):
        super(SID, self).__init__()
        self.eps = epsilon

    def forward(self, inp, target):
        normalize_inp = (inp / torch.sum(inp, dim=0)) + self.eps
        normalize_tar = (target / torch.sum(target, dim=0)) + self.eps
        sid = torch.sum(normalize_inp * torch.log(normalize_inp / normalize_tar) +
                        normalize_tar * torch.log(normalize_tar / normalize_inp))

        return sid
