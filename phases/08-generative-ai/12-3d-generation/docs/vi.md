# Thế hệ 3D

> 3D là phương thức mà đòn bẩy 2D sang 3D mạnh nhất. Bước đột phá năm 2023 là 3D Gaussian Splatting. Các lớp đẩy tổng quát 2024-2026 khuếch tán đa chế độ xem + tái tạo 3D ở trên cùng để tạo ra các đối tượng và cảnh từ một prompt hoặc ảnh duy nhất.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 4 (Tầm nhìn), Giai đoạn 8 · 07 (Khuếch tán tiềm ẩn)
**Thời lượng:** ~45 phút

## Vấn đề

Nội dung 3D rất đau đớn:

- **Đại diện.** Lưới, đám mây điểm, lưới voxel, trường khoảng cách có dấu (SDF), trường bức xạ thần kinh (NeRF), Gaussian 3D. Mỗi người đều có sự đánh đổi.
- **Khan hiếm dữ liệu.** ImageNet có 14 triệu hình ảnh. dataset 3D sạch lớn nhất (Objaverse-XL, 2023) có ~10 triệu đối tượng, hầu hết chất lượng thấp.
- **Bộ nhớ.** Lưới voxel 512³ là 128 triệu voxel; một cảnh hữu ích NeRF cần 1M samples/ray. Thế hệ khó hơn tái tạo.
- **Giám sát.** Đối với hình ảnh 2D, bạn có các pixel. Đối với 3D, bạn thường có một số chế độ xem 2D và phải nâng lên 3D.

stack 2026 tách biệt hai vấn đề. Đầu tiên, tạo *hình ảnh đa chế độ xem 2D* với model khuếch tán. Thứ hai, phù hợp với *biểu diễn 3D* (thường là bắn tung tóe Gaussian) cho những hình ảnh đó.

## Khái niệm

![3D generation: multi-view diffusion + 3D reconstruction](../assets/3d-generation.svg)

### Đại diện: 3D Gaussian Splatting (Kerbl và cộng sự, 2023)

Biểu diễn một cảnh dưới dạng cloud của ~1M 3D Gaussians. Mỗi loại có 59 parameters: vị trí (3), hiệp phương sai (6, hoặc quaternion 4 + thang 3), độ mờ (1), màu sóng hài hình cầu (48 ở bậc 3, 3 ở bậc 0).

Kết xuất = phép chiếu + tổng hợp alpha. Nhanh (~100 khung hình / giây ở 1080p trên 4090). Có thể phân biệt. Phù hợp với gradient descent với những bức ảnh chân thực. Một cảnh phù hợp trong 5-30 phút trên GPU tiêu dùng.

Hai cải tiến 2023-2024 đứng đầu:
- **Generative Gaussian splats.** Models như LGM, LRM, InstantMesh dự đoán cloud Gaussian trực tiếp từ một hoặc một vài hình ảnh.
- **4D Gaussian Splatting.** Gaussian với độ lệch trên mỗi khung hình cho các cảnh động.

### Khuếch tán đa chế độ xem

Fine-tune model khuếch tán hình ảnh pretrained để tạo nhiều chế độ xem nhất quán của cùng một đối tượng từ một prompt văn bản hoặc một hình ảnh. Zero123 (Liu và cộng sự, 2023), MVDream (Shi và cộng sự, 2023), SV3D (Độ ổn định, 2024), CAT3D (Google, 2024). Thường xuất ra 4-16 chế độ xem xung quanh vật thể, được nâng lên 3D thông qua Gaussian splatting hoặc NeRF.

### Chuyển văn bản thành pipelines 3D

| Model | Đầu vào | Đầu ra | Thời gian |
|-------|-------|--------|------|
| Giấc mơ kết hợp (2022) | Văn bản | NeRF qua SDS | ~1 giờ cho mỗi tài sản |
| Phép thuật3D | Văn bản | lưới + kết cấu | ~40 phút |
| Shap-E (OpenAI, 2023) | Văn bản | 3D ngầm | ~1 phút |
| SJC / Người mơ mộng sung mãn | Văn bản | NeRF / lưới | ~30 phút |
| LRM (Meta, 2023) | Hình Ảnh | máy bay ba tầng cánh | ~5 giây |
| Lưới tức thì (2024) | Hình Ảnh | lưới | ~10 giây |
| SV3D (Độ ổn định, 2024) | Hình Ảnh | quan điểm mới lạ | ~2 phút |
| CAT3D (Google, 2024) | 1-64 hình ảnh | 3D NeRF | ~1 phút |
| TripoSR (2024) | Hình Ảnh | lưới | ~1 giây |
| Lưới 4 (2025) | Văn bản + Hình ảnh | Lưới PBR | ~30 giây |
| Rodin thế hệ 1.5 (2025) | Văn bản + Hình ảnh | Lưới PBR | ~60 giây |
| Tencent Hunyuan3D 2.0 (2025) | Hình Ảnh | lưới | ~30 giây |

Hướng 2025-2026: models văn bản thành lưới trực tiếp với vật liệu PBR phù hợp với các công cụ trò chơi. Bước trung gian khuếch tán đa chế độ xem vẫn là công thức hoạt động tốt nhất cho các đối tượng chung.

### NeRF (cho ngữ cảnh)

Trường bức xạ thần kinh (Mildenhall và cộng sự, 2020). Một MLP nhỏ lấy `(x, y, z, view direction)` và xuất ra `(color, density)`. Kết xuất bằng cách tích hợp dọc theo các tia. Beats tổng hợp chế độ xem tiểu thuyết dựa trên lưới về chất lượng nhưng kết xuất chậm hơn 100-1000 lần. Được thay thế bởi Gaussian splatting cho hầu hết các mục đích sử dụng trong thời gian thực nhưng vẫn chiếm ưu thế trong nghiên cứu.

## Tự xây dựng

`code/main.py` thực hiện một đồ chơi 2D "Gaussian splatting": đại diện cho một hình ảnh mục tiêu tổng hợp (một gradient mượt mà) dưới dạng tổng của các vệt Gaussian 2D. Tối ưu hóa vị trí, màu sắc và hiệp phương sai theo gradient descent để phù hợp với mục tiêu. Bạn thấy hai thao tác cốt lõi: kết xuất chuyển tiếp (splat + alpha-composite) và phù hợp với gradient descent.

### Bước 1: 2D Gaussian splat

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### Bước 2: kết xuất bằng cách tổng các splat

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

Bắn tung tóe Gaussian 3D thực sắp xếp Gaussian theo độ sâu và alpha-composite theo thứ tự. Đồ chơi 2D của chúng tôi chỉ là tổng hợp.

### Bước 3: phù hợp với gradient descent

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## Cạm bẫy

- **Xem không nhất quán.** Nếu bạn tạo 4 chế độ xem độc lập và họ không đồng ý về cấu trúc đối tượng, thì độ phù hợp 3D sẽ bị mờ. Khắc phục: khuếch tán nhiều chế độ xem với attention được chia sẻ.
- **Ảo giác mặt sau.** Hình ảnh đơn → 3D phải phát minh ra mặt vô hình. Chất lượng rất khác nhau.
- **Vụ nổ bắn tung tóe Gaussian.** training không bị hạn chế phát triển thành 10 triệu lần bắn tung tóe và quá khớp. Mật độ hóa + phỏng đoán cắt tỉa (từ giấy gốc 3D-GS) là điều cần thiết.
- **Vấn đề cấu trúc liên kết.** Lưới từ trường ngầm (SDF) thường có lỗ hoặc tự giao nhau. Chạy remesher (ví dụ: remesh voxel của máy xay sinh tố) trước khi shipping.
- **Giấy phép dữ liệu training.** Objaverse có giấy phép hỗn hợp; Sử dụng thương mại khác nhau mỗi model.

## Ứng dụng

| Nhiệm vụ | Lựa chọn năm 2026 |
|------|-----------|
| Tái tạo cảnh từ ảnh | Bắn tung tóe Gaussian (3DGS, Gsplat, Scaniverse) |
| Đối tượng chuyển văn bản thành 3D cho trò chơi | Meshy 4 hoặc Rodin Gen-1.5 (đầu ra PBR) |
| Hình ảnh thành 3D | Hunyuan3D 2.0, TripoSR, InstantMesh |
| Tổng hợp chế độ xem tiểu thuyết từ một vài hình ảnh | CAT3D, SV3D |
| Tái tạo cảnh động | Bắn tung tóe Gaussian 4D |
| Hình đại diện / người mặc quần áo | Thế thần Gaussian, HUGS |
| Nghiên cứu / SOTA | Bất cứ điều gì đã rơi vào tuần trước |

Đối với shipping production 3D trong trò chơi hoặc pipeline thương mại điện tử: lưới PBR đầu ra Meshy 4 hoặc Rodin Gen-1.5 đi thẳng vào Unity / Unreal.

## Sản phẩm bàn giao

Tiết kiệm `outputs/skill-3d-pipeline.md`. Skill lấy một bản tóm tắt 3D (đầu vào: văn bản / một hình ảnh / một vài hình ảnh; đầu ra: lưới / splat / NeRF; cách sử dụng: kết xuất / trò chơi / VR) và đầu ra: pipeline (khuếch tán đa chế độ xem + phù hợp, hoặc model lưới trực tiếp), model cơ sở, ngân sách lặp lại, xử lý hậu cấu trúc liên kết, các kênh vật liệu cần thiết.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py` với 4, 16, 64 Gaussian. Báo cáo MSE cuối cùng so với mục tiêu.
2. **Trung bình.** Mở rộng đến Gaussian màu (RGB). Xác nhận tái tạo khớp với mẫu màu mục tiêu.
3. **Khó.** Sử dụng gsplat hoặc Nerfstudio, tái tạo một đối tượng thực từ ảnh chụp 50 ảnh. Báo cáo thời gian phù hợp và SSIM cuối cùng trên các chế độ xem được giữ lại.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Bắn tung tóe Gaussian 3D | "3DGS" | Cảnh như một cloud của Gaussian 3D; kết xuất alpha-composite có thể phân biệt. |
| NeRF | "Trường bức xạ thần kinh" | MLP xuất màu + mật độ tại điểm 3D; Render by Ray tích hợp. |
| Máy bay ba tầng cánh | "Ba máy bay 2-D" | Phân loại 3D thành ba lưới feature căn chỉnh trục 2D; rẻ hơn thể tích. |
| SDS | "Điểm số distillation sampling" | Huấn luyện model 3D bằng cách sử dụng điểm khuếch tán 2D làm gradient giả. |
| Khuếch tán đa chế độ xem | "Nhiều lượt xem cùng một lúc" | Khuếch tán model xuất ra một batch máy ảnh nhất quán views. |
| PBR | "Kết xuất dựa trên vật lý" | Vật liệu có suất phản chiếu, độ nhám, kim loại, các kênh bình thường. |
| Mật độ | "Phát triển splats" | 3DGS training heuristic: tách / sao chép splats ở các vùng gradient cao. |

## Lưu ý Production: 3D chưa có chất nền dùng chung

Không giống như hình ảnh (khuếch tán tiềm ẩn + DiT) và video (DiT không gian), 3D không có runtime thống trị duy nhất vào năm 2026. Cây quyết định production phân nhánh trên biểu diễn:

- **NeRF / ba tầng cánh.** Inference là hành trình tia + MLP chuyển tiếp cho mỗi mẫu. Kết xuất 512² yêu cầu hàng triệu MLP chuyển tiếp. Batch các mẫu tia một cách tích cực; SDPA/xformers áp dụng.
- **Khuếch tán đa chế độ xem + tái tạo LRM.** Hai giai đoạn pipeline. Giai đoạn 1 (DiT đa chế độ xem) là một server khuếch tán giống như Bài 07. Giai đoạn 2 (LRM transformer) là một one-shot forward pass trên tầm nhìn. Hồ sơ độ trễ tổng thể là "khuếch tán + one-shot" - chọn phân phối mỗi giai đoạn primitives cho phù hợp.
- **SDS / DreamFusion.** Tối ưu hóa cho mỗi tài sản, không phải inference. Xây dựng công việc, không phải trình xử lý yêu cầu.

Đối với hầu hết các sản phẩm năm 2026, câu trả lời đúng là "chạy model khuếch tán đa chế độ xem theo yêu cầu, tái tạo thành 3DGS không đồng bộ, phục vụ 3DGS để xem theo thời gian thực". Điều này phân chia khối lượng công việc một cách rõ ràng giữa GPU-inference server (nhanh) và optimizer ngoại tuyến (chậm).

## Đọc thêm

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) - NeRF.
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) - 3DGS.
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) - SDS.
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) - Zero123.
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) - khuếch tán đa chế độ xem.
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) - LRM.
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) - CAT3D.
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) - SV3D.
