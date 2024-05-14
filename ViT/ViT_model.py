from json import encoder
#from sympy import flatten
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout):
        super(PatchEmbedding, self).__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
        # 每张图片加一个cls_token, 所以是num_patches + 1
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        
        x = self.patcher(x).permute(0, 2, 1)
        # dim=1是指第二个维度num_patch拼接
        x = torch.cat((cls_tokens, x), dim=1) # x(batch_size,num_patches+1,embed_dim)
        # 每个patch和cls_token加上位置编码
        x += self.positional_embedding
        x = self.dropout(x)
        return x

class ViT(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout, num_heads, activation, num_encoders, num_classes):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, num_patches, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, 
                                                   activation=activation,
                                                   batch_first=True, norm_first=True)
        # 一个完整的transformer编码器，由多个nn.TransformerEncoderLayer组成，num_encoders是编码器的层数
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_encoders)
        self.MLP = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )
    
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder_layers(x)
        # 取cls_token的输出，x[:, 0, :]的维度是(batch_size, embed_dim)
        # 0是选择第一个特征，即cls_token
        # 再把cls_token的输出通过一个MLP，将cls_token的特征向量映射到类别数
        x = self.MLP(x[:, 0, :])
        return x
    

class FaceEncoder(nn.Module):
    def __init__(self, obj, in_channels, patch_size, embed_dim, num_patches, dropout, num_heads, activation, num_encoders, num_classes):
        super(FaceEncoder, self).__init__()

        # Initialize the ViT model
        self.vit = ViT(in_channels, patch_size, embed_dim, num_patches, dropout, num_heads, activation, num_encoders, num_classes)

        self.shape_dim = obj.shape_dim
        self.shape_idx = obj.shape_dim
        self.exp_idx = self.shape_idx + obj.exp_dim
        self.color_idx = self.exp_idx + obj.color_dim
        self.camera_idx = self.color_idx + 6
        self.light_idx = self.camera_idx + 27

    def forward(self, image):
        # Use the ViT model to extract features from the image
        outputs = self.vit(image)
        feature = outputs.last_hidden_state[:, 0, :]

        shape_param = feature[:, :self.shape_idx]
        exp_param = feature[:,self.shape_idx:self.exp_idx]
        color_param = feature[:, self.exp_idx:self.color_idx]
        camera_param = feature[:, self.color_idx:self.camera_idx] 
        sh_coef = feature[:,self.camera_idx:self.light_idx]

        return shape_param, exp_param, color_param,camera_param,sh_coef

