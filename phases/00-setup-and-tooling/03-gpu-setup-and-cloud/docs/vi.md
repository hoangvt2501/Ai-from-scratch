# GPU Thiết lập & Cloud

> Training trên CPU là tốt cho việc học. Training thực sự cần một GPU.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 0, Bài 01
**Thời lượng:** ~45 phút

## Mục tiêu học tập

- Xác minh tính khả dụng của GPU địa phương bằng `nvidia-smi` và CUDA API của PyTorch
- Cấu hình Google Colab với GPU T4 để thử nghiệm dựa trên cloud miễn phí
- Benchmark phép nhân ma trận trên CPU so với GPU và đo tốc độ
- Ước tính model lớn nhất phù hợp với VRAM của bạn bằng cách sử dụng quy tắc ngón tay cái fp16

## Vấn đề

Hầu hết các bài học trong giai đoạn 1-3 đều chạy tốt trên CPU. Nhưng một khi bạn bắt đầu training CNN, transformers hoặc LLMs (giai đoạn 4+), bạn cần GPU tăng tốc. Một cuộc chạy training mất 8 giờ trên CPU mất 10 phút trên GPU.

Bạn có ba tùy chọn: GPU địa phương, cloud GPU hoặc Google Colab (miễn phí).

## Khái niệm

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

## Tự xây dựng

### Lựa chọn 1: NVIDIA GPU địa phương

Kiểm tra xem bạn có không:

```bash
nvidia-smi
```

Cài đặt PyTorch với CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Tùy chọn 2: Google Colab

1. Chuyển đến [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Thay đổi loại runtime > T4 GPU
3. Chạy `!nvidia-smi` để xác minh

Tải sổ ghi chép từ khóa học này trực tiếp lên Colab.

### Lựa chọn 3: Cloud GPU

Đối với Lambda Labs, RunPod hoặc Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Không GPU? Không vấn đề gì.

Hầu hết các bài học đều hoạt động trên CPU. Những người cần GPU sẽ nói như vậy và bao gồm các liên kết Colab.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Xây dựng nó: GPU vs CPU benchmark

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## Bài tập

1. Chạy benchmark ở trên và so sánh CPU với GPU lần
2. Nếu bạn không có GPU, hãy chạy trên Google Colab và so sánh
3. Kiểm tra dung lượng bộ nhớ GPU bạn có và ước tính model lớn nhất mà bạn có thể phù hợp (quy tắc chung: 2 byte mỗi parameter đối với fp16)

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|----------------------|
| CUDA | "Lập trình GPU" | Nền tảng điện toán song song của NVIDIA cho phép bạn chạy mã trên GPU |
| VRAM | "GPU nhớ" | Video RAM trên GPU, tách biệt với hệ thống RAM. Giới hạn kích thước model. |
| FP16 | "Nửa precision" | Dấu phẩy động 16 bit, sử dụng một nửa bộ nhớ của FP32 với accuracy loss tối thiểu |
| Lõi Tensor | "Phần cứng ma trận nhanh" | Lõi GPU chuyên dụng để nhân ma trận, nhanh hơn 4-8 lần so với lõi thông thường |
