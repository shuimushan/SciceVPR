# --<utf-8>--


import torch
from torch import nn
from torch.nn import functional as F
from typing import Literal
from torchsummary import summary
import numpy as np
from einops import repeat


# Extract features from a Dino-v2 model
_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]

class DinoV2_self(nn.Module):
    """
        Extract features from an intermediate layer in Dino-v2
        从 Dino-v2 中的中间层提取特征
    """

    def __init__(self, model_name: _DINO_V2_MODELS, layer1: int = 39,  facet1: _DINO_FACETS = "value", use_cls=False,
                 norm_descs=True, device: str = "cuda:0", pretrained=True, out_indices=[8, 9, 10, 11]) -> None:
        """
            Parameters:
            - dino_model:   The DINO-v2 model to use
            - layer:        The layer to extract features from
            - facet:    "query", "key", or "value" for the attention
                        facets. "token" for the output of the layer.
            - use_cls:  If True, the CLS token (first item) is also
                        included in the returned list of descriptors.
                        Otherwise, only patch descriptors are used.
            - norm_descs:   If True, the descriptors are normalized
            - device:   PyTorch device to use
        """
        super().__init__()
        self.model_name = model_name.lower()  # 将大写转化为小写
        self.layer1 = layer1

        self.pretrained = pretrained  # 是否采用与训练参数
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.device = torch.device(device)
        self.vit_type: str = model_name
        self.out_indices = out_indices


        print(f'loading DINOv2 model（{self.model_name}）...')
        if 'vitg14' in self.model_name:
            self.dino_model = torch.hub.load('./models/backbones/facebookresearch_dinov2_main/dinov2', self.model_name, trust_repo=True, source='local')  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load('./weights/dinov2_vitg14_pretrain.pth'))
            if self.layer1 > 39:
                print('请确认layer的正确性！vitg14最高block层为39层')
                exit()
        elif 'vitl14' in self.model_name:
            self.dino_model = torch.hub.load('./models/backbones/facebookresearch_dinov2_main/dinov2', self.model_name, trust_repo=True, source='local')  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load('./weights/dinov2_vitl14_pretrain.pth'))
            if self.layer1 > 23:
                print('请确认layer的正确性！vitl14最高block层为23层')
                exit()
        elif 'vitb14' in self.model_name:
            self.dino_model = torch.hub.load('./models/backbones/facebookresearch_dinov2_main/dinov2', self.model_name, trust_repo=True, source='local')  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load('./weights/dinov2_vitb14_pretrain.pth'))
            if self.layer1 > 11:
                print('请确认layer的正确性！vitb14最高block层为12层')
                exit()
        elif 'vits14' in self.model_name:
            self.dino_model = torch.hub.load('./models/backbones/facebookresearch_dinov2_main/dinov2', self.model_name, trust_repo=True, source='local')  # 加载DINOv2预训练模型
            self.dino_model.load_state_dict(torch.load('./weights/dinov2_vits14_pretrain.pth'))
            if self.layer1 > 11:
                print('请确认layer的正确性！vits14最高block层为12层')
                exit()
        else:
            print(f'模型名称定义错误，请检查model_name:{self.dino_model}是否正确')


        self.dino_model = self.dino_model.to(self.device)
        if pretrained:
            self.dino_model.patch_embed.requires_grad_(False)

            for i in range(0, self.layer1 + 1):
                self.dino_model.blocks[i].requires_grad_(False)




    def forward(self, x, masks=None):

        x=self.dino_model.get_intermediate_layers(x,n=self.out_indices,reshape = True)#cls token excluded
        x_list = []
        for xi in x:
            if isinstance(xi, list):
                x_list.extend(xi)
            else:
                x_list.append(xi)
        x = x_list
        x = torch.cat(x, dim=1)
        return x



