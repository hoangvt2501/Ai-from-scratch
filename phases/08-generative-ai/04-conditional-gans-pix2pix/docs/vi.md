# GAN có điều kiện & Pix2Pix

> Mở khóa lớn đầu tiên của năm 2014-2017 là kiểm soát những gì GAN tạo ra. Đính kèm nhãn, hình ảnh hoặc câu. Pix2Pix đã thực hiện phiên bản hình ảnh và nó vẫn đánh bại mọi model chuyển văn bản thành hình ảnh chung trong các tác vụ chuyển hình ảnh thành hình ảnh hẹp.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 8 · 03 (GAN), Giai đoạn 4 · 06 (U-Net), Giai đoạn 3 · 07 (CNN)
**Thời lượng:** ~75 phút

## Vấn đề

Một GAN vô điều kiện lấy mẫu các khuôn mặt tùy ý. Hữu ích cho một bản demo, vô dụng trong production. Bạn muốn: *ánh xạ bản phác thảo thành ảnh*, *ánh xạ bản đồ thành ảnh trên không*, *ánh xạ cảnh ban ngày thành ban đêm*, *tô màu hình ảnh thang độ xám*. Trong tất cả những điều này, bạn được cung cấp một `x` hình ảnh đầu vào và phải xuất ra `y` với một số tương ứng ngữ nghĩa. Có rất nhiều `y` hợp lý mỗi `x`. Sai số bình phương trung bình làm phẳng chúng thành hỗn hợp. Một loss đối nghịch thì không, bởi vì "trông thật" là sắc nét.

GAN có điều kiện (Mirza & Osindero, 2014) thêm một điều kiện `c` như một đầu vào cho cả `G` và `D`. Pix2Pix (Isola et al., 2017) chuyên biệt về điều này: điều kiện là hình ảnh đầu vào đầy đủ, trình tạo là U-Net, bộ phân biệt là bộ phân loại * dựa trên bản vá * (PatchGAN) và loss là đối nghịch + L1. Công thức đó vượt trội hơn models văn bản thành hình ảnh từ đầu trên các miền hình ảnh hẹp ngay cả vào năm 2026 vì nó được huấn luyện trên *dữ liệu được ghép nối* - bạn có chính xác tín hiệu bạn cần.

## Khái niệm

![Pix2Pix: U-Net generator, PatchGAN discriminator](../assets/pix2pix.svg)

**Điều kiện G. **`G(x, z) → y`. Trong Pix2Pix, `z` dropout bên trong G (không có nhiễu đầu vào - Isola thấy nhiễu rõ ràng đã bị bỏ qua).

**D. **có điều kiện `D(x, y) → [0, 1]`. Đầu vào là * cặp * (điều kiện, đầu ra). Đây là sự khác biệt chính: D phải đánh giá liệu `y` có phù hợp với `x` hay không, chứ không chỉ là `y` có thật hay không.

**Máy phát U-Net.** Encoder-decoder với các kết nối bỏ qua qua nút cổ chai. Rất quan trọng đối với các tác vụ mà đầu vào và đầu ra chia sẻ cấu trúc cấp thấp (cạnh, hình bóng). Nếu không bỏ qua, chi tiết tần số cao sẽ biến mất.

**Phân biệt PatchGAN.** Thay vì xuất ra một điểm real/fake duy nhất, D xuất ra một lưới `N×N` trong đó mỗi ô đánh giá một trường tiếp nhận ~70×70 pixel. Tính trung bình. Đây là một giả định trường ngẫu nhiên của Markov: chủ nghĩa hiện thực là cục bộ. Huấn luyện nhanh hơn nhiều, ít parameters hơn, đầu ra sắc nét hơn.

**Loss.**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

Thuật ngữ L1 ổn định training và đẩy G về phía mục tiêu đã biết. L1 cho các cạnh sắc nét hơn L2 (trung bình, không phải phương tiện). `λ = 100` là mặc định của Pix2Pix.

## CycleGAN — khi bạn không có cặp

Pix2Pix cần dữ liệu `(x, y)` được ghép nối. CycleGAN (Zhu và cộng sự, 2017) bỏ yêu cầu này với chi phí của một loss bổ sung: loss *tính nhất quán của chu kỳ*. Hai máy phát điện `G: X → Y` và `F: Y → X`. Huấn luyện họ thật `F(G(x)) ≈ x` và `G(F(y)) ≈ y`. Điều này cho phép bạn dịch ngựa thành ngựa vằn, từ mùa hè sang mùa đông mà không cần các ví dụ ghép đôi.

Vào năm 2026, hình ảnh không ghép nối chủ yếu được thực hiện thông qua khuếch tán (ControlNet, IP-Adapter) thay vì CycleGAN, nhưng ý tưởng nhất quán chu kỳ vẫn tồn tại trong hầu hết mọi bài báo thích ứng miền không ghép đôi.

## Tự xây dựng

`code/main.py` triển khai một GAN có điều kiện nhỏ trên dữ liệu 1-D. Điều kiện `c` là nhãn class (0 hoặc 1). Nhiệm vụ: tạo mẫu từ phân phối có điều kiện cho class nhất định.

### Bước 1: thêm điều kiện vào cả đầu vào G và D

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

Mã hóa một nóng là cách đơn giản nhất. models lớn hơn sử dụng embeddings đã học, điều chế FiLM hoặc cross-attention.

### Bước 2: huấn luyện có điều kiện

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

Trình tạo phải khớp với phân phối thực *cho điều kiện nhất định*, không phải cận biên.

### Bước 3: xác minh đầu ra trên mỗi class

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## Cạm bẫy

- **Điều kiện bị bỏ qua.** G học cách gạt ra ngoài lề, D không bao giờ phạt vì tín hiệu điều kiện yếu. Khắc phục: điều kiện D tích cực hơn (lớp sớm, không chỉ muộn), sử dụng bộ phân biệt chiếu (Miyato & Koyama 2018).
- **Trọng lượng L1 quá thấp.** G trôi dạt đến đầu ra trông thật tùy ý, không phải những đầu ra trung thực. Bắt đầu λ≈100 cho các tác vụ kiểu Pix2Pix.
- **Trọng lượng L1 quá cao.** G tạo ra đầu ra mờ vì L1 vẫn là một tiêu chuẩn L_p. Ủ xuống khi training ổn định.
- **Rò rỉ sự thật cơ bản trong D. **Nối `(x, y)` làm đầu vào D, không chỉ `y`. Nếu không có điều này, D không thể kiểm tra tính nhất quán.
- **Thu gọn chế độ trên mỗi class.** Mỗi class có thể thu gọn độc lập. Chạy kiểm tra đa dạng class điều kiện.

## Ứng dụng

Trạng thái năm 2026 của các tác vụ từ hình ảnh thành hình ảnh:

| Nhiệm vụ | Cách tiếp cận tốt nhất |
|------|---------------|
| Phác thảo ảnh →, cùng tên miền, dữ liệu được ghép nối | Pix2Pix / Pix2PixHD (vẫn nhanh, vẫn sắc nét) |
| Phác thảo ảnh →, chưa ghép nối | ControlNet với model điều hòa Scribble |
| Seg ngữ nghĩa → ảnh | SPADE / GauGAN2 hoặc SD + ControlNet-Seg |
| Chuyển đổi phong cách | Khuếch tán với IP-Adapter hoặc LoRA; Phương pháp GAN là kế thừa |
| Độ sâu → ảnh | ControlNet-Depth trên khuếch tán ổn định |
| Siêu phân giải | Real-ESRGAN (GAN), ESRGAN-Plus hoặc SD-Upscale (khuếch tán) |
| Màu sắc | ColTran, chất tạo màu dựa trên khuếch tán hoặc Pix2Pix-color |
| Ban ngày → ban đêm, mùa, thời tiết | CycleGAN hoặc ControlNet-based |

Pix2Pix vẫn là công cụ phù hợp khi (a) bạn có hàng nghìn ví dụ được ghép nối, (b) nhiệm vụ hẹp và có thể lặp lại, và (c) bạn cần inference nhanh. Trên các tác vụ miền mở chung, khuếch tán chiến thắng.

## Sản phẩm bàn giao

Tiết kiệm `outputs/skill-img2img-chooser.md`. Skill lấy mô tả tác vụ, tính khả dụng của dữ liệu (ghép nối và không ghép nối, N mẫu) và ngân sách latency/quality, sau đó xuất ra: cách tiếp cận (Pix2Pix, CycleGAN, biến thể ControlNet, SDXL + IP-Adapter), training yêu cầu dữ liệu, chi phí inference và giao thức đánh giá (LPIPS, FID, nhiệm vụ cụ thể).

## Bài tập

1. **Dễ dàng.** Sửa đổi `code/main.py` để thêm class thứ ba. Xác nhận G vẫn ánh xạ nhiễu của từng class sang chế độ chính xác.
2. **Trung bình.** Thay thế L1 bằng loss kiểu tri giác trong cài đặt 1-D (ví dụ: một D nhỏ đóng băng hoạt động như bộ trích xuất feature). Nó có thay đổi độ sắc nét của phân phối có điều kiện không?
3. **Khó.** Phác thảo một CycleGAN trong cài đặt 1-D: hai phân phối, hai máy phát điện, chu kỳ loss. Cho thấy rằng nó học cách ánh xạ giữa chúng mà không có dữ liệu được ghép nối.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| GAN có điều kiện | "GAN có nhãn" | G(z, c), D(x, c). Cả hai mạng đều nhìn thấy tình trạng này. |
| Pix2Pix | "GAN từ hình ảnh sang hình ảnh" | Ghép nối cGAN với U-Net G và PatchGAN D + L1 loss. |
| Mạng U-Net | "Encoder-decoder có bỏ qua" | Mạng conv đối xứng; bỏ qua duy trì tần số cao. |
| Bản vá GAN | "Bộ phân loại hiện thực cục bộ" | D xuất ra điểm cho mỗi bản vá thay vì điểm toàn cục. |
| CycleGAN | "Bản dịch hình ảnh chưa ghép nối" | Hai G + tính nhất quán của chu kỳ loss; Không có dữ liệu được ghép nối. |
| BÍCH | "GauGAN" | Chuẩn hóa các hoạt động trung gian với bản đồ ngữ nghĩa; phân đoạn thành hình ảnh. |
| Hồ sơ | "Điều chế tuyến tính Feature khôn ngoan" | Mỗi feature affine biến đổi từ điều kiện; điều hòa giá rẻ. |

## Lưu ý Production: Pix2Pix làm đường cơ sở giới hạn độ trễ

Khi bạn đã ghép nối dữ liệu và một nhiệm vụ hẹp (phác thảo → kết xuất, bản đồ ngữ nghĩa → ảnh, ngày → đêm), one-shot inference của Pix2Pix đánh bại sự khuếch tán theo thứ tự độ lớn về độ trễ. So sánh production thường là:

| Đường dẫn | Các bước | Độ trễ điển hình ở 512² trên một L4 |
|------|-------|----------------------------------------|
| Pix2Pix (Chuyển tiếp U-Net) | 1 | ~30 mili giây |
| SD-Inpaint hoặc SD-Img2Img | 20 | ~1,2 giây |
| SDXL-Turbo Img2Img | 1-4 | ~0,15-0,35 giây |
| Cơ sở ControlNet + SDXL | 20-30 | ~3-5 giây |

Pix2Pix giành chiến thắng về thông lượng trong batches tĩnh (mọi yêu cầu đều giống nhau FLOPs). Khuếch tán chiến thắng về chất lượng và khái quát hóa. Trò chơi hiện đại thường là ship một model chưng cất theo phong cách Pix2Pix cho nhiệm vụ hẹp và dự phòng khuếch tán cho đầu vào đuôi.

## Đọc thêm

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) - bài báo cGAN.
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) - Pix2Pix.
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) - CycleGAN.
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) - Pix2PixHD.
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) - SPADE / GauGAN.
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — hình chiếu D.
