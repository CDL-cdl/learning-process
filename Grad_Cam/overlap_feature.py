#grad_cam 叠加多个target layer
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from grad_cam_util import GradCAM, show_cam_on_image
from models.networks import define_net_recon


def main():
    # 初始化模型
    model = define_net_recon(net_recon='resnet50', use_last_fc=False, init_path="../../checkpoints/deep3d/bottle_co-downaff/epoch_34.pth")
    target_layers = model.backbone.fusion[-1]  # 使用最后的融合层

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

    # 初始化 Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layers], use_cuda=False)

    # 定义一个空的叠加热力图
    combined_grayscale_cam = np.zeros((img_tensor.shape[1], img_tensor.shape[2]))

    # 计算 0 到 79 的 Grad-CAM 并叠加
    for target_category in range(80, 143):
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        combined_grayscale_cam += grayscale_cam[0, :]  # 叠加每个任务的热力图

    # 归一化叠加后的热力图
    combined_grayscale_cam /= 64  # 平均

    # 将热力图叠加到原始图像上
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      combined_grayscale_cam,
                                      use_rgb=True)

    # 保存叠加后的 Grad-CAM 输出
    output_img_path = "grad_cam_output.jpg"
    plt.imsave(output_img_path, visualization)
    print(f"Grad-CAM output saved to {output_img_path}")


if __name__ == '__main__':
    main()