import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import cv2  # 添加OpenCV库用于调整大小
from torchvision import transforms
from models.networks import define_net_recon
from grad_cam_util import show_cam_on_image

def extract_feature_maps(model, input_tensor, layers):
    """ 提取指定层的特征图 """
    feature_maps = []
    hooks = []

    def hook_fn(module, input, output):
        feature_maps.append(output)

    # 注册hook函数，在指定的层上提取特征图
    for layer in layers:
        hooks.append(layer.register_forward_hook(hook_fn))

    # 前向传播，生成特征图
    _ = model(input_tensor)

    # 移除hook函数
    for hook in hooks:
        hook.remove()

    return feature_maps

def main():
    # 初始化模型
    model = define_net_recon(net_recon='resnet50', use_last_fc=False, init_path="../../checkpoints/deep3d/pre-trained/epoch_20.pth")
    print(model)
    model.eval()  # 切换到评估模式

    # 定义要提取特征图的层，比如特征融合层（fusion层）
    target_layers = [model.backbone.fusion[0], model.backbone.fusion[1], model.backbone.fusion[2]]

    # 定义数据预处理步骤
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # 加载图像
    img_path = "./datasets/examples/000002.jpg"
    assert os.path.exists(img_path), f"file: '{img_path}' does not exist."
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)

    # 预处理图像
    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)  # 扩展 batch 维度

    # 提取指定层的特征图
    feature_maps = extract_feature_maps(model, input_tensor, target_layers)

    # 可视化每个层的特征图
    for i, fmap in enumerate(feature_maps):
        fmap = fmap[0]  # 去除batch维度
        fmap = fmap.detach().cpu().numpy()

        # 将特征图的所有通道平均，并归一化到0-1之间
        fmap_avg = np.mean(fmap, axis=0)
        fmap_avg = (fmap_avg - np.min(fmap_avg)) / (np.max(fmap_avg) - np.min(fmap_avg))

        # 调整特征图大小以匹配输入图像的大小
        fmap_avg_resized = cv2.resize(fmap_avg, (img.shape[1], img.shape[0]))

        # 将特征图叠加到原始图像上
        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255., fmap_avg_resized, use_rgb=True)

        # 保存叠加后的输出图像
        output_img_path = f"feature_map_layer_{i}.jpg"
        plt.imsave(output_img_path, visualization)
        print(f"Feature map visualization saved to {output_img_path}")

if __name__ == '__main__':
    main()
