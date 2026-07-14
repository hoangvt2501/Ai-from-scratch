# Tầm nhìn Transformers (ViT)

> Hình ảnh là một lưới các bản vá. Một câu là một lưới tokens. Cùng một transformer ăn cả hai.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 7 · 05 (Transformer đầy đủ), Giai đoạn 4 · 03 (CNN), Giai đoạn 4 · 14 (Giới thiệu Transformers tầm nhìn)
**Thời lượng:** ~45 phút

## Vấn đề

Trước năm 2020, thị giác máy tính có nghĩa là tích chập. Mọi SOTA trên ImageNet, COCO và benchmarks phát hiện đều sử dụng đường trục CNN. Transformers dành cho ngôn ngữ.

Dosovitskiy et al. (2020) - "Một hình ảnh có giá trị 16x16 từ" - cho thấy bạn có thể bỏ hoàn toàn các chập chập. Cắt hình ảnh thành các bản vá có kích thước cố định, chiếu tuyến tính từng bản vá thành một embedding, đưa trình tự vào transformer encoder vani. Ở quy mô đủ (ImageNet-21k pretraining hoặc lớn hơn), ViT phù hợp hoặc đánh bại models dựa trên ResNet.

ViT là sự khởi đầu của một mô hình rộng lớn hơn vào năm 2026: một kiến trúc, nhiều phương thức. Whisper mã hóa âm thanh. ViT mã hóa hình ảnh. tokens hành động cho robot. Pixel tokens cho video. Người transformer không quan tâm - cho nó ăn một trình tự và nó học hỏi.

Đến năm 2026, ViT và hậu duệ của nó (DeiT, Swin, DINOv2, ViT-22B, SAM 3) sở hữu hầu hết tầm nhìn. CNN vẫn giành chiến thắng trên các thiết bị biên và các tác vụ nhạy cảm với độ trễ. Mọi thứ khác đều có ViT ở đâu đó trong stack.

## Khái niệm

![Image → patches → tokens → transformer](../assets/vit.svg)

### Bước 1 - vá lỗi

Chia hình ảnh `H × W × C` thành một chuỗi `N × (P·P·C)` các mảng phẳng. Thiết lập điển hình: `224 × 224` hình ảnh, `16 × 16` bản vá → 196 bản vá, mỗi bản vá 768 giá trị.

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

Kích thước bản vá là đòn bẩy. Các bản vá nhỏ hơn = nhiều tokens hơn, độ phân giải tốt hơn, chi phí attention bậc hai. Các bản vá lớn hơn = thô hơn, rẻ hơn.

### Bước 2 - embedding tuyến tính

Một ma trận đã học duy nhất chiếu mỗi mảng phẳng đến `d_model`. Tương đương với sự tích chập của kích thước hạt nhân `P` và sải chân `P`. Trong PyTorch điều này thực sự `nn.Conv2d(C, d_model, kernel_size=P, stride=P)` - một triển khai 2 dòng.

### Bước 3 - thêm `[CLS]` token, thêm embeddings vị trí

- Thêm vào trước một `[CLS]` token có thể học được. Trạng thái ẩn cuối cùng của nó là biểu diễn hình ảnh được sử dụng để phân loại.
- Thêm embeddings vị trí có thể học được (ViT-gốc) hoặc hình sin 2D (các biến thể sau này).
- Vào năm 2024+ RoPE mở rộng sang 2D cho vị trí, đôi khi không có embeddings rõ ràng.

### Bước 4 - transformer encoder tiêu chuẩn

Stack L khối `LayerNorm → Self-Attention → + → LayerNorm → MLP → +`. Giống hệt với BERT. Không có lớp dành riêng cho tầm nhìn. Đây là điểm nhấn sư phạm của bài báo.

### Bước 5 - đầu

Để phân loại: lấy trạng thái ẩn `[CLS]` → → softmax tuyến tính. Đối với DINOv2 hoặc SAM, hãy loại bỏ `[CLS]`, sử dụng embeddings bản vá trực tiếp.

### Các biến thể quan trọng

| Model | Năm | Thay đổi |
|-------|------|--------|
| ViT | 2020 | Bản gốc. Kích thước bản vá cố định, attention toàn cầu đầy đủ. |
| DeiT | 2021 | Distillation; chỉ có thể huấn luyện trên ImageNet-1k. |
| Swin | 2021 | Phân cấp với windows dịch chuyển. Chi phí bậc hai phụ cố định. |
| DINOv2 | 2023 | Tự giám sát (không có nhãn). Thị lực chung tốt nhất features. |
| ViT-22B | 2023 | 22 tỷ tham số; luật tỷ lệ được áp dụng. |
| SigLIP | 2023 | Cặp ngôn ngữ ViT +, loss tương phản sigmoid. |
| SAM 3 | 2025 | Phân đoạn bất cứ thứ gì; ViT-Large + mặt nạ nhanh chóng decoder. |

### Tại sao phải mất một thời gian

ViT cần * rất nhiều * dữ liệu để khớp với CNN vì nó không có thiên vị quy nạp CNN (bất biến dịch thuật, địa phương). Nếu không có >100 triệu hình ảnh được gắn nhãn hoặc pretraining tự giám sát mạnh mẽ, CNN vẫn giành chiến thắng ở mức điện toán phù hợp. DeiT đã khắc phục điều này vào năm 2021 bằng distillation thủ thuật; DINOv2 đã khắc phục nó vĩnh viễn vào năm 2023 với tính năng tự giám sát.

## Tự xây dựng

Xem `code/main.py`. Pure-stdlib patchify + embedding tuyến tính + kiểm tra sự tỉnh táo. Không training - ViT ở bất kỳ quy mô thực tế nào cần PyTorch và hàng giờ GPU.

### Bước 1: hình ảnh giả mạo

Hình ảnh 24 × 24 RGB dưới dạng danh sách các hàng `(R, G, B)` bộ dữ liệu. Chúng ta sử dụng 6×6 bản vá → 16 bản vá, 108-d embedding vector mỗi bản vá.

### Bước 2: vá lỗi

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

Thứ tự raster: hàng trưởng trên lưới. Mỗi ViT đều sử dụng thứ tự này.

### Bước 3: nhúng tuyến tính

Nhân mỗi mảng phẳng với một ma trận `(patch_flat_size, d_model)` ngẫu nhiên. Xác minh hình dạng đầu ra là `(N_patches + 1, d_model)` sau khi thêm `[CLS]`.

### Bước 4: đếm parameters để có ViT thực tế

In số lượng tham số cho ViT-Base: 12 lớp, 12 đầu, d = 768, bản vá = 16. So sánh với ResNet-50 (~25M). ViT-Base hạ cánh ở ~86M. ViT-Lớn ~307M. ViT-Khổng lồ ~632M.

## Ứng dụng

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 embeddings là mặc định năm 2026 cho features hình ảnh.** Đóng băng xương sống, huấn luyện một cái đầu nhỏ. Hoạt động để phân loại, truy xuất, phát hiện, chú thích. DINOv2 của Meta checkpoints vượt trội hơn CLIP trên mọi tác vụ thị giác không phải văn bản.

**Chọn kích thước miếng dán.** Nhỏ models sử dụng 16×16 (ViT-B/16). Dự đoán dày đặc (phân đoạn) sử dụng 8×8 hoặc 14×14 (SAM, DINOv2). Rất lớn models sử dụng 14×14.

## Sản phẩm bàn giao

Xem `outputs/skill-vit-configurator.md`. skill chọn biến thể ViT và kích thước bản vá cho một tác vụ tầm nhìn mới dựa dataset kích thước, độ phân giải và ngân sách tính toán.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py`. Xác minh số lượng bản vá bằng `(H/P) * (W/P)` và kích thước bản vá phẳng bằng `P*P*C`.
2. **Trung bình.** Triển khai embeddings vị trí hình sin 2D — hai mã hình sin độc lập cho `row` và `col` của mỗi bản vá, được nối với nhau. Cung cấp chúng vào một PyTorch ViT nhỏ và so sánh embeddings vị trí accuracy và có thể học được trên CIFAR-10.
3. **Khó.** Xây dựng ViT 3 lớp (PyTorch), huấn luyện trên 1.000 hình ảnh MNIST với 4×4 bản vá. Đo lường kiểm tra accuracy. Bây giờ thêm pretraining DINOv2 trên cùng 1.000 hình ảnh (đơn giản hóa: chỉ cần huấn luyện encoder để dự đoán embeddings bản vá từ các bản vá được che mặt). accuracy có cải thiện không?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Bản vá | "Tầm nhìn-transformer token" | vector phẳng của các giá trị pixel cho một vùng `P × P × C` của hình ảnh. |
| Vá lỗi | "Chặt + dẹt" | Cắt hình ảnh thành các mảng không chồng lên nhau, làm phẳng mỗi mảng thành một vector. |
| `[CLS]` token | "Tóm tắt hình ảnh" | token có thể học được trước đó; embedding cuối cùng của nó là biểu diễn hình ảnh. |
| bias quy nạp | "Những gì model giả định" | ViT có ít priors hơn CNN; cần nhiều dữ liệu hơn để bù đắp khoảng trống. |
| DINOv2 | "ViT tự giám sát" | Được huấn luyện không có nhãn bằng cách sử dụng giáo viên tăng cường hình ảnh + động lượng. features hình ảnh chung đẹp nhất năm 2026. |
| SigLIP | "Người kế nhiệm CLIP" | Văn bản ViT + encoder huấn luyện với loss tương phản sigmoid; tốt hơn CLIP trên điện toán phù hợp. |
| Swin | "ViT có cửa sổ" | ViT phân cấp với attention cục bộ + dịch chuyển windows; bậc hai phụ. |
| Đăng ký tokens | "Thủ thuật năm 2023" | Một vài tokens có thể học thêm thấm attention bồn rửa; cải thiện features DINOv2. |

## Đọc thêm

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) - bài báo ViT.
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT.
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) - Swin.
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) - DINOv2.
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — bản sửa lỗi thanh ghi token cho DINOv2.
