import torch
import numpy as np
from PIL import Image
import sys
import os

# 将当前目录添加到路径以便导入 pipelines
sys.path.append(os.getcwd())

def test_environment():
    print("正在检查环境依赖...")
    
    try:
        import lpips
        print("✅ lpips 模块导入成功")
    except ImportError:
        print("❌ 缺少 lpips 模块，请运行: pip install lpips")
        return

    try:
        from pytorch_msssim import ssim
        print("✅ pytorch_msssim 模块导入成功")
    except ImportError:
        print("❌ 缺少 pytorch-msssim 模块，请运行: pip install pytorch-msssim")
        return

    print("\n正在测试梯度计算函数...")
    
    try:
        from pipelines.similarity_gradient import (
            compute_ssim_gradient,
            compute_lpips_gradient,
            create_lpips_model
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # 创建假数据
        img1 = Image.new('RGB', (64, 64), color='red')
        img2 = Image.new('RGB', (64, 64), color='blue')
        
        # 测试 SSIM
        print("测试 compute_ssim_gradient...", end=" ")
        ssim_res = compute_ssim_gradient(img1, img2, device=device)
        # ssim_grad: [3, 64, 64]
        if ssim_res['ssim_grad'].shape == (3, 64, 64):
            print("✅ 成功")
        else:
            print(f"❌ 形状错误: {ssim_res['ssim_grad'].shape}")

        # 测试 LPIPS
        print("测试 compute_lpips_gradient (这可能需要下载模型)...", end=" ")
        lpips_model = create_lpips_model(device)
        lpips_res = compute_lpips_gradient(img1, img2, lpips_model, device=device)
        # lpips_grad: [3, 64, 64]
        if lpips_res['lpips_grad'].shape == (3, 64, 64):
            print("✅ 成功")
        else:
            print(f"❌ 形状错误: {lpips_res['lpips_grad'].shape}")
            
        print("\n所有基础环境测试通过！")

    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_environment()

