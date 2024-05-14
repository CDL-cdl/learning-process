## from https://github.com/lucidrains/vit-pytorch
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads   # 64 * 8 = 512

        #多头注意力的头数不为1，或每个头的维度不等于期望的维度，那么就需要对输出进行投影。
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5   # 64 ** -0.5 = 0.125

        self.attend = nn.Softmax(dim = -1)  # 对最后一个维度进行softmax
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),  # 512 -> 1024
            nn.Dropout(dropout) 
        ) if project_out else nn.Identity()

    def forward(self, x):
        # x → (query, key, value)
        # (1,197,3072) → (1,197,1024) (1,197,1024) (1,197,1024)
        qkv = self.to_qkv(x).chunk(3, dim = -1)## 对tensor张量分块 x :1 197 1024   qkv 最后是一个元祖，tuple，长度是3，每个元素形状：1 197 1024
        # map将rearrange函数应用于qkv元组中的每个元素,(batch_size, heads, seq_length, depth)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # 1 8 197 64

        # 计算q,k的点积(q,k的匹配程度), * 缩放因子
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # 计算注意力权重
        attn = self.attend(dots)

        # 加权求和
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # PreNorm + Multi-head attention 
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                # PreNorm + FeedForward
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size) ## 224*224
        patch_height, patch_width = pair(patch_size)## 16 * 16

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width) 
        patch_dim = channels * patch_height * patch_width   # Flatten_dim = 3 * 16 * 16 = 768
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Flatten + Projection
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),  # 768 -> 1024
        )
        
        # num_patches + 1 是因为有一个额外的cls token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # 在训练中更新cls_token
        # 每个输入图像都有一个额外的cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # img 1 3 224 224  
        #输出形状x : 1 196 1024  |(224/16)*(224/16) = 196
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape ## 

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)



v = ViT(
    image_size = 224,
    patch_size = 16,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 224, 224)

preds = v(img) # (1, 1000)