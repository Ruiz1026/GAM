import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import to_2tuple
class PatchEmbed(nn.Module):
    def __init__(self, img_size=14, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size)*(img_size[1] // patch_size)
        self.proj = nn.Sequential(nn.Conv2d(in_chans,embed_dim, kernel_size=patch_size, stride=patch_size),
                                  nn.BatchNorm2d(embed_dim),
                                     nn.ReLU(inplace=True))
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops
class PatchUnEmbed(nn.Module):

    def __init__(self, img_size=14, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size)*(img_size[1] // patch_size)
        self.unproj = nn.Sequential(nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=patch_size, stride=patch_size),
                                    nn.BatchNorm2d(in_chans),
                                     nn.ReLU(inplace=True))
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size, x_size)  # B Ph*Pw C
        x = self.unproj(x)
        return x

    def flops(self):
        flops = 0
        return flops
class CrossAdaptiveAttention(nn.Module):
    def __init__(self,img_size=14,num_head=8,patch_size=4,in_chans=2,dim=512,dropout_pro=0.0):
        super().__init__()
        self.patch_embed=PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=dim,
            norm_layer=nn.LayerNorm)
        self.patch_unembed=PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=dim,
            norm_layer=nn.LayerNorm)
        self.img_size=img_size
        self.patch_size=patch_size
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.attn = torch.nn.MultiheadAttention(embed_dim = dim, num_heads = num_head, dropout = dropout_pro)
    def forward(self,teacher,vae):
        teacher_embbed=self.patch_embed(teacher)
        vae_embbed=self.patch_embed(vae)
        q=self.W_q(vae_embbed)
        k=self.W_k(teacher_embbed)
        v=self.W_v(teacher_embbed)
        attn_outputs,attn_weights=self.attn(q,k,v)
        res=self.patch_unembed(attn_outputs,self.img_size // self.patch_size)
        return res

class GCALoss(nn.Module):  
    def __init__(self,img_size=16,num_head=8,patch_size=4,in_chans=256,dim=256,dropout_pro=0.0,gaze_channels=3):
        super().__init__()
        self.CA=CrossAdaptiveAttention(img_size=16,num_head=8,patch_size=4,in_chans=256,dim=512,dropout_pro=0.0)

        self.in_block=nn.Sequential(nn.Conv2d(3, 128, kernel_size=1,padding=0),nn.Conv2d(128, 64, kernel_size=1,padding=0))
        self.align=nn.Sequential(nn.Conv2d(64, 128, kernel_size=3,stride=2,padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 256, kernel_size=3,stride=2,padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=3,stride=2,padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=3,stride=2,padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True),)
        self.enc=nn.Sequential(nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(in_chans),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1),nn.BatchNorm2d(in_chans),nn.ReLU(inplace=True))
        self.dec =nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.gcaLoss= nn.MSELoss(reduction='mean')
    def forward(self,t,s,gaze):
        t=self.enc(t)
        gaze=self.in_block(gaze)
        gaze_prompt=self.align(gaze)
        x=self.CA(t,gaze_prompt)
        #x=nn.functional.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)
        tg=torch.cat((x,t),dim=1)
        tg=self.dec(tg)
        loss_gca=self.gcaLoss(s,tg)
        return loss_gca
