import os
import pickle
from natten.functional import na2d_av
from einops import rearrange, einsum
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
import plots
import transformer
import utils
from SSFE import FeatureExtractor
from FMBA import FMBA
from decoder import MultiEndBranchAttention

class AutoEncoder(nn.Module):
    def __init__(self, P, L, size, patch, dim):
        super(AutoEncoder, self).__init__()
        self.P, self.L, self.size, self.dim = P, L, size, dim

        self.c5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c6 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.c7 = nn.Conv2d(64, 64, kernel_size=7, padding=3)
        self.c8 = nn.Conv2d(64, 64, kernel_size=9, padding=4)
        self.b5 = nn.BatchNorm2d(64, momentum=0.9)

        self.spatial_spectral_feature = FeatureExtractor(L)

        self.CDDP2 = CDDP(dim=64, ctx_dim=64, kernel_size=7, smk_size=5, num_heads=2)
        self.activation = nn.ReLU()
        self.fusion_module = FusionModule(64,P)

        self.vtrans = transformer.ViT(image_size=size, patch_size=patch, dim=(dim*P), depth=2,
                                      heads=8, mlp_dim=1, pool='cls')
        self.upscale = nn.Sequential(
            nn.Linear(dim, size ** 2),
        )
        self.smooth = SmoothBlock(P)

        self.decoder = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=(1, 1), stride=(1, 1), bias=False),
        )
        self.end_branch_att = MultiEndBranchAttention(n_end=P, hidden_dim=dim)
        self.fmba = FMBA(in_channels=L)


    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x_K =x
        print(x_K.shape)

        feq_features = self.fmba(x_K)
        spatial_features, spectral_features = self.spatial_spectral_feature(x_K)

        cat_features1 = torch.cat((spatial_features,feq_features), dim=1)
        cat_features2 = torch.cat((spatial_features,spectral_features), dim=1)
        features1 = self.CDDP2(cat_features1)
        features2 = self.CDDP2(cat_features2)

        test_features = self.c5(cat_features1)
        test_features = self.b5(test_features)
        test_features = self.c6(test_features)
        test_features = self.b5(test_features)
        test_features = self.c7(test_features)
        test_features = self.b5(test_features)
        test_features = self.c8(test_features)
        test_features = self.b5(test_features)

        test_features1 = self.c5(cat_features2)
        test_features1 = self.b5(test_features1)
        test_features1 = self.c6(test_features1)
        test_features1 = self.b5(test_features1)
        test_features1 = self.c7(test_features1)
        test_features1 = self.b5(test_features1)
        test_features1 = self.c8(test_features1)
        test_features1 = self.b5(test_features1)

        features = features1 + 0.2*self.activation(test_features)
        feq = features2 + 0.2*self.activation(test_features1)

        abu_est = self.fusion_module(features,feq)


        cls_emb = self.vtrans(abu_est)
        cls_emb = cls_emb.view(1, self.P, -1)


        end_sem = cls_emb.squeeze(0)
        end_sem = self.end_branch_att(end_sem)


        end_sem = end_sem.unsqueeze(0)
        abu_est = self.upscale(end_sem)
        abu_est = abu_est.view(1, self.P, self.size, self.size)
        abu_est = self.smooth(abu_est)
        re_result = self.decoder(abu_est)

        return abu_est, re_result

class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(5e-3, 0.99999)
class Train_test:
    def __init__(self, dataset, device, skip_train=False, save=False):
        super(Train_test, self).__init__()
        self.skip_train = skip_train
        self.device = device
        self.dataset = dataset
        self.save = save
        self.save_dir = "trans_mod_" + dataset + "/"
        os.makedirs(self.save_dir, exist_ok=True)

        if dataset == 'sy30A':
            self.P, self.L, self.col = 5, 200, 60
            self.LR, self.EPOCH = 1e-3, 400
            self.patch, self.dim = 5, 200
            self.step_size, self.a = 20, 0.90
            self.beta, self.gamma = 8e2, 1e-2
            self.weight_decay_param = 2e-5
            self.order_abd, self.order_endmem = (0, 1, 2, 3,4), (0, 1, 2, 3,4)
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=self.col ** 2)
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
        elif dataset == 'DC3_20':
            self.P, self.L, self.col = 6, 224, 105
            self.LR, self.EPOCH = 1.5e-3, 500  # 1.5e-3 500
            self.patch, self.dim = 5, 200
            self.step_size, self.a = 20, 0.95  # 20，0.95
            self.beta, self.gamma = 9e4, 5e-4  # 9e4, 5e-4
            self.weight_decay_param = 5e-6  # 5e-6
            self.order_abd, self.order_endmem = (0, 2, 5, 1, 4, 3), (0, 2, 5, 1, 4, 3)
            self.data = datasets.Data(dataset, device)
            self.loader = self.data.get_loader(batch_size=self.col ** 2)
            self.init_weight = self.data.get("init_weight").unsqueeze(2).unsqueeze(3).float()
        else:
            raise ValueError("Unknown dataset")

        # ===== Build network once so that it can be accessed outside =====
        self.net = AutoEncoder(P=self.P, L=self.L, size=self.col,
                               patch=self.patch, dim=self.dim).to(self.device)

    def run(self, smry):
        net = self.net
        if smry:
            print(net)
            return

        net.apply(net.weights_init)
        model_dict = net.state_dict()
        model_dict['decoder.0.weight'] = self.init_weight
        net.load_state_dict(model_dict)


        loss_func = nn.MSELoss(reduction='mean')

        loss_func2 = utils.SAD(self.L)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.LR, weight_decay=self.weight_decay_param)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.a)
        apply_clamp_inst1 = NonZeroClipper()

        if not self.skip_train:
            net.train()
            epo_vs_los = []
            for epoch in range(self.EPOCH):
                for i, (x, _) in enumerate(self.loader):

                    x= x.transpose(1, 0).view(1, -1, self.col, self.col)
                    abu_est, re_result = net(x)
                    loss_re = self.beta * loss_func(re_result, x)
                    loss_sad = loss_func2(re_result.view(1, self.L, -1).transpose(1, 2),
                                              x.view(1, self.L, -1).transpose(1, 2))
                    loss_sad = self.gamma * torch.sum(loss_sad).float()
                    total_loss = loss_re + loss_sad
                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
                    optimizer.step()  # 更新参数
                    net.decoder.apply(apply_clamp_inst1)
                    if epoch % 1 == 0:
                        print('Epoch:', epoch, '| train loss: %.4f' % total_loss.data,
                              '| re loss: %.4f' % loss_re.data,
                              '| sad loss: %.4f' % loss_sad.data)
                    epo_vs_los.append(float(total_loss.data))

                    with open(self.save_dir + 'weights_new.pickle', 'wb') as handle:
                        pickle.dump(net.state_dict(), handle)

                scheduler.step()


        else:
            with open(self.save_dir + 'weights_new.pickle', 'rb') as handle:
                net.load_state_dict(pickle.load(handle))

        # Testing ================
        net.eval()
        x = self.data.get("hs_img").transpose(1, 0).view(1, -1, self.col, self.col)
        abu_est, re_result = net(x)
        abu_est = abu_est / (torch.sum(abu_est, dim=1))
        abu_est = abu_est.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


        target = torch.reshape(self.data.get("abd_map"), (self.col, self.col, self.P)).cpu().numpy()


        true_endmem = self.data.get("end_mem").numpy()


        est_endmem = net.state_dict()["decoder.0.weight"].cpu().numpy()
        est_endmem = est_endmem.reshape((self.L, self.P))


        abu_est = abu_est[:, :, self.order_abd]
        est_endmem = est_endmem[:, self.order_endmem]

        sio.savemat(self.save_dir + f"{self.dataset}_abd_map.mat", {"[]]": abu_est})
        sio.savemat(self.save_dir + f"{self.dataset}_endmem.mat", {"E_est": est_endmem})

        x = x.view(-1, self.col, self.col).permute(1, 2, 0).detach().cpu().numpy()
        re_result = re_result.view(-1, self.col, self.col).permute(1, 2, 0).detach().cpu().numpy()
        re = utils.compute_re(x, re_result)
        print("RE:", re)


        rmse_cls, mean_rmse = utils.compute_rmse(target, abu_est)
        print(target.shape)
        print("Class-wise RMSE value:")
        for i in range(self.P):
            print("Class", i + 1, ":", rmse_cls[i])
        print("Mean RMSE:", mean_rmse)


        sad_cls, mean_sad = utils.compute_sad(est_endmem, true_endmem)
        print("Class-wise SAD value:")
        for i in range(self.P):
            print("Class", i + 1, ":", sad_cls[i])
        print("Mean SAD:", mean_sad)

        with open(self.save_dir + "log1.csv", 'a') as file:
            file.write(f"LR: {self.LR}, ")
            file.write(f"WD: {self.weight_decay_param}, ")
            file.write(f"B: {self.beta}, ")
            file.write(f"RE: {re:.4f}, ")
            file.write(f"SAD: {mean_sad:.4f}, ")
            file.write(f"RMSE: {mean_rmse:.4f}\n")



        for i in range(self.P):
            est_endmem[:, i] = est_endmem[:, i] / est_endmem[:, i].max()
            true_endmem[:, i] = true_endmem[:, i] / true_endmem[:, i].max()
        plots.plot_abundance(target, abu_est, self.P, self.save_dir)
        plots.plot_endmembers(true_endmem, est_endmem, self.P, self.save_dir)
        plots.plt.legend()
        plots.plt.show()
        
# =================================================================
class CDDP(nn.Module):
    def __init__(self,
                 dim=64,
                 ctx_dim=32,
                 kernel_size=7,
                 smk_size=5,
                 num_heads=4,
                 ):
        super().__init__()
        ctx_dim = dim // 2
        self.kernel_size = kernel_size
        self.smk_size = smk_size
        self.num_heads = num_heads * 2
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.weight_query = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 2, kernel_size=1, bias=False),  # 32 -> 32
            nn.BatchNorm2d(dim // 2),
        )

        self.weight_key = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(ctx_dim, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 2),
        )

        self.weight_proj = nn.Conv2d(49, kernel_size ** 2 + smk_size ** 2, kernel_size=1)

        self.dyconv_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.get_rpb()

    def get_rpb(self):
        self.rpb_size1 = 2 * self.smk_size - 1
        self.rpb1 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size1, self.rpb_size1))
        self.rpb_size2 = 2 * self.kernel_size - 1
        self.rpb2 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size2, self.rpb_size2))
        nn.init.zeros_(self.rpb1)
        nn.init.zeros_(self.rpb2)

    @torch.no_grad()
    def generate_idx(self, kernel_size):
        rpb_size = 2 * kernel_size - 1
        idx_h = torch.arange(0, kernel_size)
        idx_w = torch.arange(0, kernel_size)
        idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
        return (idx_h, idx_w, idx_k)

    def apply_rpb(self, attn, rpb, height, width, kernel_size, idx_h, idx_w, idx_k):
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_h[kernel_size // 2] = height - (kernel_size - 1)
        num_repeat_w[kernel_size // 2] = width - (kernel_size - 1)
        bias_hw = (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (
                    2 * kernel_size - 1)) + idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + idx_k
        bias_idx = bias_idx.reshape(-1, int(kernel_size ** 2))
        bias_idx = torch.flip(bias_idx, [0])
        rpb = torch.flatten(rpb, 1, 2)[:, bias_idx]
        rpb = rpb.reshape(1, int(self.num_heads), int(height), int(width), int(kernel_size ** 2))
        return attn + rpb

    def forward(self,x):
        B, C, H, W = x.shape

        query, key = torch.chunk(x, 2, dim=1)  # 32, 32
        query = self.weight_query(query) * self.scale
        key = self.weight_key(key)
        query = rearrange(query, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        key = rearrange(key, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        weight = einsum(query, key, 'b g c n, b g c l -> b g n l')
        weight = rearrange(weight, 'b g n l -> b l g n').contiguous()
        weight = self.weight_proj(weight)
        weight = rearrange(weight, 'b l g (h w) -> b g h w l', h=H, w=W)

        attn1, attn2 = torch.split(weight, split_size_or_sections=[self.smk_size ** 2, self.kernel_size ** 2], dim=-1)
        rpb1_idx = self.generate_idx(self.smk_size)
        rpb2_idx = self.generate_idx(self.kernel_size)
        attn1 = self.apply_rpb(attn1, self.rpb1, H, W, self.smk_size, *rpb1_idx)
        attn2 = self.apply_rpb(attn2, self.rpb2, H, W, self.kernel_size, *rpb2_idx)
        attn1 = torch.softmax(attn1, dim=-1)
        attn2 = torch.softmax(attn2, dim=-1)
        value = rearrange(x, 'b (m g c) h w -> m b g h w c', m=2, g=self.num_heads)

        x1 = na2d_av(attn1, value[0], kernel_size=self.smk_size)
        x2 = na2d_av(attn2, value[1], kernel_size=self.kernel_size)

        x = torch.cat([x1, x2], dim=1)
        x = rearrange(x, 'b g h w c -> b (g c) h w', h=H, w=W)
        x = self.dyconv_proj(x)
        return x

class LearnableSoftmax(nn.Module):
    def __init__(self,
                 dim=-1,
                 init_temperature=1.0,
                 min_temp=0.80,
                 max_temp=1.15):

        super().__init__()
        self.dim = dim

        self.min_temp = min_temp
        self.max_temp = max_temp

        init_temperature = max(init_temperature, 0.8)
        self.log_temp = nn.Parameter(
            torch.log(torch.tensor([init_temperature], dtype=torch.float32))
        )

    def forward(self, x):
        temperature = torch.clamp(
            torch.exp(self.log_temp),
            self.min_temp,
            self.max_temp
        )
        x_scaled = x / temperature
        x_max, _ = torch.max(x_scaled, dim=self.dim, keepdim=True)
        x_exp = torch.exp(x_scaled)
        x_sum = torch.sum(x_exp, dim=self.dim, keepdim=True)
        return x_exp / x_sum
class FusionModule(nn.Module):
    def __init__(self, ctx_dim,P):
        super().__init__()
        self.x_scale = LayerScale(ctx_dim, init_value=1)
        self.h_scale = LayerScale(ctx_dim, init_value=1)
        self.mlp = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim),
            nn.GELU()
        )
        self.c=nn.Conv2d(ctx_dim, P*8, kernel_size=1, padding=0)

    def forward(self, x, h):
        # 展平 h 为 (1*95*95, 64) -> Linear -> 恢复为 (1, 64, 95, 95)
        h_flat = h.permute(0, 2, 3, 1).reshape(-1, h.shape[1])  # [9025, 64]
        h_transformed = self.mlp(h_flat)  # [9025, 64]
        h_transformed = h_transformed.view(x.shape[0], x.shape[2], x.shape[3], -1)
        h = h_transformed.permute(0, 3, 1, 2)  # [1, 64, 95, 95]
        # 特征融合

        return self.c(self.x_scale(x) + self.h_scale(h))



class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1)*init_value,
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x

class SmoothBlock(nn.Module):
    def __init__(self, P):
        super(SmoothBlock, self).__init__()
        # 主卷积序列
        self.conv_layers = nn.Sequential(
            #nn.Conv2d(P, P, kernel_size=(11,11), stride=(1, 1), padding=(5, 5)),
            nn.Conv2d(P, P, kernel_size=(9,9),  stride=(1,1),  padding=(4,4)),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(P, P, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(P, P, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(P, P, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            #nn.Conv2d(P, P, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),

        )
        self.softmax = LearnableSoftmax(dim=1, init_temperature=0.95)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 主卷积计算
        out = self.conv_layers(x)

        out = self.softmax(out)
        return out



