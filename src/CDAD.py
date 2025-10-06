# basic
import numpy as np
import timm
import re
import copy
import sys
import math
import random
import os
import wget

# torch
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim

# image processing
from sklearn.metrics import roc_auc_score, auc
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from cv2 import getStructuringElement, MORPH_RECT, dilate

from timm.models.layers import DropPath

# focal loss
from utils.focal_loss import FocalLoss

def my_forward_wrapper(attn_obj):
    def my_forward(x):
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x

    return my_forward


class FeatureExtractor_ViT(nn.Module):
    def __init__(self, hf_path, layer_indices, image_size, device):
        super(FeatureExtractor_ViT, self).__init__()
        self.layer_indices = layer_indices

        # get pretrained model
        self.pretrained_model = timm.create_model(hf_path, pretrained=True, num_classes=0).to(device)

        # freeze the pretrained model's parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # store shapes
        self.embed_dim = 1024
        self.patch_size = 14
        self.num_patches = (image_size // self.patch_size) ** 2

        # wrapper to be able to extract the attention map
        self.pretrained_model.blocks[layer_indices[-1] - 1].attn.forward = my_forward_wrapper(
            self.pretrained_model.blocks[layer_indices[-1] - 1].attn)

    def forward(self, x, output_attn=False):
        # initial input processing
        features = self.pretrained_model.forward_features(x)
        features = features[:, 1:, :]

        if output_attn:
            attn_map = self.pretrained_model.blocks[self.layer_indices[-1] - 1].attn.attn_map
            attn_map_cls = attn_map[:, :, 0, 1:]

        return (features, attn_map_cls) if output_attn else features


class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.layer_norm_1(x + self.dropout1(attn_output))
        x = x + self.dropout2(self.linear(x))
        return x


class ConvBN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        """构造一个包含卷积与批归一化的基本层。"""
        super().__init__()
        # 若未指定填充，自动设置为卷积核尺寸的一半；若k为列表，则对每个元素进行相同计算。
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        # 初始化一个不使用偏置的二维卷积层
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        # 初始化批归一化层以稳定训练过程
        self.bn = nn.BatchNorm2d(c2)
        # 采用SiLU激活函数（注意：当前前向传播版本未使用激活）
        self.act = nn.SiLU()

    def forward(self, x):
        """对输入张量执行卷积和批归一化操作。"""
        return self.bn(self.conv(x))  # 仅执行卷积和BN，不包括激活。  # ai缝合大王


class Conv_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        """
        Conv_Block模块通过深度卷积与1x1卷积扩展通道，
        结合ReLU6非线性激活和残差连接，来实现特征的重分配。
        """
        super().__init__()
        # 采用7x7深度卷积，其中groups设为dim实现逐通道卷积
        self.dwconv = ConvBN(dim, dim, 7, g=dim)
        # 两个1x1卷积分别用于将特征扩展至mlp_ratio倍的通道数
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        # 通过1x1卷积整合扩展后的特征
        self.g = ConvBN(mlp_ratio * dim, dim, 1)
        # 使用另一个7x7深度卷积实现进一步的特征重分配
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)
        # ReLU6激活函数限制输出范围
        self.act = nn.ReLU6()
        # 引入DropPath机制来随机丢弃部分路径以缓解过拟合；当drop_path=0时，该层为恒等映射
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """前向传播：提取特征后进行非线性变换，最后通过残差连接融合输入。"""
        input = x  # 保存原始输入用于残差连接
        x = self.dwconv(x)  # 通过7x7深度卷积提取局部特征
        x1, x2 = self.f1(x), self.f2(x)  # 通过两个1x1卷积扩展特征通道
        x = self.act(x1) * x2  # 将激活后的第一分支与第二分支逐元素相乘
        x = self.g(x)  # 使用1x1卷积融合特征
        x = self.dwconv2(x)  # 再次应用7x7深度卷积细化特征
        x = input + self.drop_path(x)  # 添加残差连接并应用DropPath机制
        return x


class FeatureAdaptor(nn.Module):
    def __init__(self, embed_dim):
        super(FeatureAdaptor, self).__init__()
        self.linear = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        input = x
        x = x.contiguous().view(-1, embed_dim)
        x = self.linear(x)
        x = x.view(batch_size, num_patches, -1)
        x = x + input
        return x


class MFFD(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_patches, num_layers=1, num_heads=12, dropout_rate=0):
        super(MFFD, self).__init__()

        # Transformer encoder layer
        self.transformer_encoder = nn.Sequential(
            *[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout_rate) for _ in range(num_layers)])

        # Output layer
        self.output_layer = nn.Linear(embed_dim, 1, bias=False)

        # Learnable positional encodings
        self.positional_encodings = nn.Parameter(torch.randn(num_patches, embed_dim))

        self.block = Conv_Block(dim=1024)

    def forward(self, x):
        # pos emb
        x = x + self.positional_encodings.unsqueeze(0)
        x = self.transformer_encoder(x)
        transformer_output = x

        x = x.reshape(-1, 37, 37, 1024)
        x = x.permute(0, 3, 1, 2)
        x = self.block(x)  # x.shape=16,1024,37,37
        x = x.reshape(-1, 1024, 37 * 37)
        x = x.transpose(1, 2)

        # 残差连接
        x = x + transformer_output
        x = self.output_layer(x).squeeze(-1)
        return x


class CDAD(pl.LightningModule):

    def __init__(self, lr, lr_decay_factor, lr_adaptor, hf_path, layers_to_extract_from, hidden_dim, wd, epochs,
                 noise_std,
                 dsc_layers, dsc_heads, dsc_dropout, pool_size, image_size, num_fake_patches, fake_feature_type, top_k,
                 log_pixel_metrics, smoothing_sigma, smoothing_radius):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # 这里改动了

        self.feature_extractor = FeatureExtractor_ViT(hf_path, layers_to_extract_from, image_size, self.device)
        self.attn_output = True

        # get feature extractor parameters
        embed_dim = self.feature_extractor.embed_dim
        num_patches = self.feature_extractor.num_patches
        self.patch_size = self.feature_extractor.patch_size
        self.patches_per_side = int(np.sqrt(num_patches))

        if top_k < 0 or top_k > num_patches:
            self.top_k = num_patches
        else:
            self.top_k = top_k

        # make sure the number of fake patches doesnt crash the code
        if num_fake_patches < 0 or num_fake_patches > num_patches:
            self.num_fake_patches = num_patches
        else:
            self.num_fake_patches = num_fake_patches

        # models
        self.discriminator = MFFD(embed_dim, hidden_dim, num_patches, dsc_layers, dsc_heads,
                                               dsc_dropout).to(self.device)
        self.fa = FeatureAdaptor(embed_dim).to(self.device)  # 改动

        # focal loss
        self.focal_loss = FocalLoss().to(self.device)

        # init for evaluation
        self.val_scores = []
        self.val_labels = []
        self.val_masks = []
        self.test_scores = []
        self.test_labels = []
        self.test_masks = []

    def forward(self, x):
        scores = self._step(x)
        return scores

    def configure_optimizers(self):
        optimizer_dsc = optim.AdamW(self.discriminator.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)

        optimizer_fa = optim.AdamW(self.fa.parameters(), lr=self.hparams.lr_adaptor)  # 改动

        lr_scheduler_dsc = optim.lr_scheduler.CosineAnnealingLR(optimizer_dsc, self.hparams.epochs,
                                                                self.hparams.lr * self.hparams.lr_decay_factor)

        lr_scheduler_fa = optim.lr_scheduler.LambdaLR(optimizer_fa, lr_lambda=lambda epoch: 1)  # 改动

        return [
            {  # 判别器优化组
                "optimizer": optimizer_dsc,
                "lr_scheduler": {
                    "scheduler": lr_scheduler_dsc,
                    "interval": "epoch"  # 每个epoch更新
                    # 可添加其他参数如frequency/monitor等
                }
            },
            {  # fa模块优化组
                "optimizer": optimizer_fa,
                "lr_scheduler": {
                    "scheduler": lr_scheduler_fa,
                    "interval": "epoch"
                }
            }
        ]

    def _step(self, images):
        features = self.feature_extractor(images)
        features = self.fa(features)  # 改动
        scores = self.discriminator(features)
        return scores

    def _add_random_noise(self, features):
        # clone features to avoid in-place modification
        fake_features = features.clone()

        # create a noise tensor
        noise = torch.normal(0, self.hparams.noise_std, features.shape).to(self.device)

        batch_size, num_patches, feature_dim = features.shape
        masks = torch.zeros((batch_size, num_patches), dtype=torch.bool)

        for i in range(batch_size):
            num_fake = random.randint(1, num_patches)
            random_indices = torch.randperm(num_patches)[:num_fake]
            masks[i, random_indices] = True    # 使用花式索引
            fake_features[i, random_indices, :] += noise[i, random_indices, :]

        return fake_features, masks

    def _add_attn_noise(self, features, attn_map):
        # clone features to avoid in-place modification
        fake_features = features.clone()

        # create a noise tensor
        noise = torch.normal(0, self.hparams.noise_std, features.shape).to(self.device)

        batch_size, num_heads, num_patches = attn_map.shape
        masks = torch.zeros((batch_size, num_patches), dtype=torch.bool)

        for i in range(batch_size):
            head = random.randint(0, num_heads - 1)
            num_fake = random.randint(1, num_patches)
            max_attn_indices = torch.tensor(list(torch.topk(attn_map[i, head, :], num_fake)[1]))
            masks[i, max_attn_indices] = True
            fake_features[i, max_attn_indices, :] += noise[i, max_attn_indices, :]

        return fake_features, masks

    def _add_attn_copy_out(self, features, attn_map):
        # clone features to avoid in-place modification
        fake_features = features.clone()

        batch_size, num_heads, num_patches = attn_map.shape
        masks = torch.zeros((batch_size, num_patches), dtype=torch.bool)

        for i in range(batch_size):
            # take the number of patches with the highest attention score from a random chosen head
            head = random.randint(0, num_heads - 1)
            num_fake = random.randint(1, self.num_fake_patches)
            max_attn_indices = torch.tensor(list(torch.topk(attn_map[i, head, :], num_fake)[1]))
            random_indices = torch.randperm(num_patches)[:num_fake]
            masks[i, max_attn_indices] = True
            fake_features[i, max_attn_indices, :] = features[i, random_indices, :]

        return fake_features, masks

    def _add_attn_shuffle(self, features, attn_map):
        # clone features to avoid in-place modification
        fake_features = features.clone()

        batch_size, num_heads, num_patches = attn_map.shape
        masks = torch.zeros((batch_size, num_patches), dtype=torch.bool)

        for i in range(batch_size):
            # take the number of patches with the highest attention score from a random chosen head
            head = random.randint(0, num_heads - 1)
            num_fake = random.randint(1, self.num_fake_patches)
            max_attn_indices = torch.tensor(list(torch.topk(attn_map[i, head, :], num_fake)[1]))
            masks[i, max_attn_indices] = True

            # shuffle
            shuffled_patches = fake_features[i, max_attn_indices].clone()
            shuffled_patches = shuffled_patches[torch.randperm(num_fake)]
            fake_features[i, max_attn_indices, :] = shuffled_patches

        return fake_features, masks

    def _add_random_shuffle(self, features):
        # clone features to avoid in-place modification
        fake_features = features.clone()

        batch_size, num_patches, feature_dim = features.shape
        masks = torch.zeros((batch_size, num_patches), dtype=torch.bool)

        for i in range(batch_size):
            # take the number of patches with the highest attention score from a random chosen head
            num_fake = random.randint(1, self.num_fake_patches)
            random_indices = torch.randperm(num_patches)[:num_fake]
            masks[i, random_indices] = True

            # shuffle
            shuffled_patches = fake_features[i, random_indices].clone()
            shuffled_patches = shuffled_patches[torch.randperm(num_fake)]
            fake_features[i, random_indices, :] = shuffled_patches

        return fake_features, masks

    def _add_noise_all(self, features):
        # clone features to avoid in-place modification
        fake_features = features.clone()

        # create a noise tensor
        noise = torch.normal(0, self.hparams.noise_std, features.shape).to(self.device)
        fake_features = fake_features + noise
        masks = torch.ones(features.shape[0] * features.shape[1]).to(self.device)

        return fake_features, masks

    def training_step(self, batch, batch_idx):
        images = batch[0]  # 取到一个batch的图片

        if self.attn_output:  # attn_output为true，则返回注意力图
            features, attn_map = self.feature_extractor(images, output_attn=True)
            features = self.fa(features)  # 改动
        else:
            features = self.feature_extractor(images)
            features = self.fa(features)  # 改动

        # stack loss
        loss = 0

        fake_features, masks_fake = self._add_noise_all(features)
        scores_true = self.discriminator(features).flatten()
        masks_true = torch.zeros(features.shape[0] * features.shape[1]).to(self.device)
        loss += F.binary_cross_entropy_with_logits(scores_true, masks_true)
        loss += self.focal_loss(scores_true, masks_true)

        scores_fake = self.discriminator(fake_features)
        loss += F.binary_cross_entropy_with_logits(scores_fake.flatten(), masks_fake.flatten())
        loss += self.focal_loss(scores_fake.flatten(), masks_fake.flatten())

        if self.hparams.fake_feature_type in ['random', 'copy_out_and_random', 'shuffle_and_random',
                                              'randshuffle_and_random']:
            # add noise to random subset of the patches
            random_features, masks_random = self._add_random_noise(features)
            scores_random = self.discriminator(random_features).flatten()
            masks_random = masks_random.flatten().to(self.device)
            loss += F.binary_cross_entropy_with_logits(scores_random[~masks_random],
                                                       masks_random[~masks_random].float())
            loss += self.focal_loss(scores_random[~masks_random], masks_random[~masks_random].float())
            loss += F.binary_cross_entropy_with_logits(scores_random[masks_random], masks_random[masks_random].float())
            loss += self.focal_loss(scores_random[masks_random], masks_random[masks_random].float())
        elif self.hparams.fake_feature_type in ['attn', 'copy_out_and_attn', 'shuffle_and_attn',
                                                'randshuffle_and_attn']:
            # add noise to patches with highest attention in the feature extractor
            attn_features, masks_attn = self._add_attn_noise(features, attn_map)
            scores_attn = self.discriminator(attn_features).flatten()
            masks_attn = masks_attn.flatten().to(self.device)
            loss += F.binary_cross_entropy_with_logits(scores_attn[~masks_attn], masks_attn[~masks_attn].float())
            loss += self.focal_loss(scores_attn[~masks_attn], masks_attn[~masks_attn].float())
            loss += F.binary_cross_entropy_with_logits(scores_attn[masks_attn], masks_attn[masks_attn].float())
            loss += self.focal_loss(scores_attn[masks_attn], masks_attn[masks_attn].float())

        if self.hparams.fake_feature_type in ['copy_out', 'copy_out_and_random', 'copy_out_and_attn']:
            # perform cutpaste in the feature space based on the patches with the highest attention value
            copy_features, masks_copy = self._add_attn_copy_out(features, attn_map)
            scores_copy = self.discriminator(copy_features).flatten()
            masks_copy = masks_copy.flatten().to(self.device)
            loss += F.binary_cross_entropy_with_logits(scores_copy[~masks_copy], masks_copy[~masks_copy].float())
            loss += self.focal_loss(scores_copy[~masks_copy], masks_copy[~masks_copy].float())
            loss += F.binary_cross_entropy_with_logits(scores_copy[masks_copy], masks_copy[masks_copy].float())
            loss += self.focal_loss(scores_copy[masks_copy], masks_copy[masks_copy].float())
        elif self.hparams.fake_feature_type in ['shuffle', 'shuffle_and_random', 'shuffle_and_attn']:
            # shuffle the patches with the highest attention values in the feature extractor
            shuffle_features, masks_shuffle = self._add_attn_shuffle(features, attn_map)
            scores_shuffle = self.discriminator(shuffle_features).flatten()
            masks_shuffle = masks_shuffle.flatten().to(self.device)
            loss += F.binary_cross_entropy_with_logits(scores_shuffle[~masks_shuffle],
                                                       masks_shuffle[~masks_shuffle].float())
            loss += self.focal_loss(scores_shuffle[~masks_shuffle], masks_shuffle[~masks_shuffle].float())
            loss += F.binary_cross_entropy_with_logits(scores_shuffle[masks_shuffle],
                                                       masks_shuffle[masks_shuffle].float())
            loss += self.focal_loss(scores_shuffle[masks_shuffle], masks_shuffle[masks_shuffle].float())
        elif self.hparams.fake_feature_type in ['randshuffle', 'randshuffle_and_random', 'randshuffle_and_attn']:
            # shuffle a random subset of the patches
            randshuffle_features, masks_randshuffle = self._add_random_shuffle(features)
            scores_randshuffle = self.discriminator(randshuffle_features).flatten()
            masks_randshuffle = masks_randshuffle.flatten().to(self.device)
            loss += F.binary_cross_entropy_with_logits(scores_randshuffle[~masks_randshuffle],
                                                       masks_randshuffle[~masks_randshuffle].float())
            loss += self.focal_loss(scores_randshuffle[~masks_randshuffle],
                                    masks_randshuffle[~masks_randshuffle].float())
            loss += F.binary_cross_entropy_with_logits(scores_randshuffle[masks_randshuffle],
                                                       masks_randshuffle[masks_randshuffle].float())
            loss += self.focal_loss(scores_randshuffle[masks_randshuffle], masks_randshuffle[masks_randshuffle].float())

        self.log('train_loss', loss)

        # Manual optimization
        loss.backward()
        opt_dsc, opt_fa = self.optimizers()
        opt_dsc.step()
        opt_fa.step()
        opt_fa.zero_grad()
        opt_dsc.zero_grad()

        return loss

    def on_train_epoch_end(self, unused=None):
        lr_schedulers = self.lr_schedulers()
        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()

    def validation_step(self, batch, batch_idx):
        images = batch[0]
        labels = batch[1]
        scores = self._step(images)
        self.val_scores.append(scores)
        self.val_labels.append(labels)

        if self.hparams.log_pixel_metrics:
            masks = batch[2]
            self.val_masks.append(masks)

    def test_step(self, batch, batch_idx):
        images = batch[0]
        labels = batch[1]
        scores = self._step(images)
        self.test_scores.append(scores)
        self.test_labels.append(labels)

        if self.hparams.log_pixel_metrics:
            masks = batch[2]
            self.test_masks.append(masks)

    def on_validation_epoch_end(self):
        scores = torch.cat(self.val_scores, dim=0)

        topk_values, _ = torch.topk(scores, self.top_k, dim=1)
        image_scores = torch.mean(topk_values, dim=1)
        image_labels = torch.cat(self.val_labels, dim=0)

        # calculate I-AUROC
        image_auroc = roc_auc_score(image_labels.view(-1).cpu().numpy(), image_scores.view(-1).cpu().numpy())
        self.log('val_image_auroc', round(image_auroc, 3), on_epoch=True)

        if self.hparams.log_pixel_metrics:
            masks = torch.cat(self.val_masks, dim=0)
            patch_scores = scores.reshape(-1, self.patches_per_side, self.patches_per_side)
            pixel_scores = F.interpolate(patch_scores.unsqueeze(1), size=(masks.shape[-1], masks.shape[-1]),
                                         mode='bilinear', align_corners=False)
            segmentations = gaussian_filter(pixel_scores.squeeze(1).cpu().detach().numpy(),
                                            sigma=self.hparams.smoothing_sigma, radius=self.hparams.smoothing_radius,
                                            axes=(1, 2))

            # calculate P-AUROC
            pixel_auroc = roc_auc_score(masks.view(-1).cpu().numpy(), segmentations.reshape(-1))
            self.log('val_pixel_auroc', round(pixel_auroc, 3), on_epoch=True)

        self.val_scores = []
        self.val_labels = []
        self.val_masks = []

    def _compute_pro(self, masks, segmentations, num_th=200):
        binary_segmentations = np.zeros_like(segmentations, dtype=bool)

        min_th = segmentations.min()
        max_th = segmentations.max()
        delta = (max_th - min_th) / num_th

        patch = getStructuringElement(MORPH_RECT, (self.patch_size, self.patch_size))

        pro_data = []
        fpr_data = []

        for th in np.arange(min_th, max_th, delta):
            binary_segmentations[segmentations <= th] = 0
            binary_segmentations[segmentations > th] = 1

            pros = []
            for binary_segmentation, mask in zip(binary_segmentations, masks):
                binary_segmentation = dilate(binary_segmentation.astype(np.uint8), patch)
                for region in regionprops(label(mask)):
                    x_idx = region.coords[:, 0]
                    y_idx = region.coords[:, 1]
                    tp_pixels = binary_segmentation[x_idx, y_idx].sum()
                    pros.append(tp_pixels / region.area)

            inverse_masks = 1 - masks
            fp_pixels = np.logical_and(inverse_masks, binary_segmentations).sum()
            fpr = fp_pixels / inverse_masks.sum()

            pro_data.append(np.mean(pros))
            fpr_data.append(fpr)

        fpr_data = np.array(fpr_data)
        pro_data = np.array(pro_data)

        valid_indices = fpr_data < 0.3
        fpr_data = fpr_data[valid_indices]
        pro_data = pro_data[valid_indices]

        fpr_data = fpr_data / fpr_data.max()

        aupro = auc(fpr_data, pro_data)
        return aupro

    def on_test_epoch_end(self):
        scores = torch.cat(self.test_scores, dim=0)

        topk_values, _ = torch.topk(scores, self.top_k, dim=1)
        image_scores = torch.mean(topk_values, dim=1)
        image_labels = torch.cat(self.test_labels, dim=0)

        # calculate I-AUROC
        image_auroc = roc_auc_score(image_labels.view(-1).cpu().numpy(), image_scores.view(-1).cpu().numpy())
        self.log('test_image_auroc', round(image_auroc, 3), on_epoch=True)

        if self.hparams.log_pixel_metrics:
            masks = torch.cat(self.test_masks, dim=0)
            patch_scores = scores.reshape(-1, self.patches_per_side, self.patches_per_side)
            pixel_scores = F.interpolate(patch_scores.unsqueeze(1), size=(masks.shape[-1], masks.shape[-1]),
                                         mode='bilinear', align_corners=False)
            segmentations = gaussian_filter(pixel_scores.squeeze(1).cpu().detach().numpy(),
                                            sigma=self.hparams.smoothing_sigma, radius=self.hparams.smoothing_radius,
                                            axes=(1, 2))

            # calculate P-AUROC
            pixel_auroc = roc_auc_score(masks.view(-1).cpu().numpy(), segmentations.reshape(-1))
            self.log('test_pixel_auroc', round(pixel_auroc, 3), on_epoch=True)

            # calculate PRO-score
            pro_score = self._compute_pro(masks.cpu().numpy(), segmentations)
            self.log('test_pro_score', round(pro_score, 3), on_epoch=True)

        self.test_scores = []
        self.test_labels = []
        self.test_masks = []
