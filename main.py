import torch
import torchvision

# 檢查是否支持 CUDA
print("CUDA is available:", torch.cuda.is_available())

# 顯示 cuDNN 版本
cudnn_version = torch.backends.cudnn.version()
print(f"cuDNN version: {cudnn_version if cudnn_version else 'Not available'}")

if torch.cuda.is_available():
    # 在 CUDA 上創建張量
    a = torch.zeros(2, device='cuda')
    print(f"Tensor a: {a}")

    b = torch.randn(2, device='cuda')
    print(f"Tensor b: {b}")

    c = a + b
    print(f"Tensor c: {c}")

    # 顯示 CUDA 設備信息
    print("Number of CUDA devices:", torch.cuda.device_count())
    print("Current CUDA device index:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("No CUDA device detected. Cannot proceed with GPU operations.")

# 顯示 torchvision 版本
print(f"TorchVision version: {torchvision.__version__}")

