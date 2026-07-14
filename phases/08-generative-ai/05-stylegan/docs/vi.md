# Phong cáchGAN

> Hầu hết các máy phát điện đều khuấy `z` vào mọi lớp cùng một lúc. StyleGAN tách nó ra: đầu tiên ánh xạ `z` đến một `w` trung gian, sau đó *tiêm* `w` ở mọi cấp độ phân giải thông qua AdaIN. Sự thay đổi duy nhất đó đã gỡ rối không gian tiềm ẩn và làm cho khuôn mặt chân thực trở thành một vấn đề được giải quyết trong bảy năm liên tiếp.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 8 · 03 (GAN), Giai đoạn 4 · 08 (Chuẩn hóa), Giai đoạn 3 · 07 (CNN)
**Thời lượng:** ~45 phút

## Vấn đề

DCGAN ánh xạ `z` đến một hình ảnh thông qua một stack các tích chập chuyển vị. Vấn đề: `z` kiểm soát mọi thứ - tư thế, ánh sáng, bản sắc, bối cảnh - vướng víu với nhau. Di chuyển dọc theo một trục `z`, cả bốn đều thay đổi. Bạn không thể hỏi model "cùng một người, tư thế khác nhau" vì đại diện không tính theo cách đó.

Karras et al. (2019, NVIDIA) đề xuất: ngừng cho `z` ăn trực tiếp vào các lớp conviz. Cung cấp một `4×4×512` tensor không đổi làm đầu vào mạng. Tìm hiểu MLP 8 lớp ánh xạ `z ∈ Z → w ∈ W`. Chèn `w` ở mọi độ phân giải thông qua *chuẩn hóa phiên bản thích ứng* (AdaIN): chuẩn hóa từng bản đồ feature chuyển đổi, sau đó chia tỷ lệ và dịch chuyển bằng phép chiếu afin của `w`. Thêm nhiễu trên mỗi lớp cho chi tiết ngẫu nhiên (lỗ chân lông trên da, sợi tóc).

Kết quả: `W` có các trục gần như trực giao cho "phong cách cấp cao" (tư thế, nhận dạng) so với "phong cách đẹp" (ánh sáng, màu sắc). Bạn có thể hoán đổi kiểu giữa hai hình ảnh bằng cách sử dụng `w` của hình ảnh A cho mức độ phân giải thấp và `w` của hình ảnh B cho mức độ phân giải cao. Điều này đã mở khóa chỉnh sửa, cách điệu đa miền và toàn bộ dòng nghiên cứu "Đảo ngược StyleGAN".

## Khái niệm

![StyleGAN: mapping network + AdaIN + per-layer noise](../assets/stylegan.svg)

**Mạng lập bản đồ.** `f: Z → W`, một MLP 8 lớp. `Z = N(0, I)^512`. `W` không bị ép buộc phải là Gaussian - nó học một hình dạng thích ứng với dữ liệu.

**Mạng tổng hợp.** Bắt đầu từ một `4×4×512` không đổi đã học. Mỗi khối độ phân giải: `upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`. Độ phân giải gấp đôi: 4, 8, 16, 32, 64, 128, 256, 512, 1024.

**AdaIN.**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

trong đó `y_scale` và `y_bias` đến từ các phép chiếu affine của `w`. Chuẩn hóa mỗi feature bản đồ, sau đó tạo kiểu lại. "Phong cách" ở đây là số liệu thống kê bậc một và thứ hai của bản đồ feature.

**Nhiễu trên mỗi lớp.** Nhiễu Gaussian một kênh được thêm vào mỗi bản đồ feature, được chia tỷ lệ theo hệ số trên mỗi kênh đã học. Kiểm soát chi tiết ngẫu nhiên mà không ảnh hưởng đến cấu trúc toàn cầu.

**Thủ thuật cắt bớt.** Tại inference, lấy mẫu `z`, tính toán `w = mapping(z)`, sau đó `w' = ŵ + ψ·(w - ŵ)` trong đó `ŵ` là giá trị trung bình `w` trên nhiều mẫu. `ψ < 1` đánh đổi sự đa dạng để lấy chất lượng. Hầu hết mọi bản demo StyleGAN đều sử dụng `ψ ≈ 0.7`.

## Phong cáchGAN 1 → 2 → 3

| Phiên bản | Năm | Đổi mới |
|---------|------|------------|
| Phong cáchGAN | 2019 | Mạng lập bản đồ + AdaIN + nhiễu + phát triển dần dần. |
| Phong cáchGAN2 | 2020 | Giải điều chế trọng lượng thay thế AdaIN (sửa lỗi giọt artifacts); skip/residual kiến trúc; chính quy hóa chiều dài đường dẫn. |
| Phong cáchGAN3 | 2021 | Tích chập không có bí danh + hạt nhân biến đều; Loại bỏ kết cấu dính vào lưới pixel. |
| Phong cáchGAN-XL | 2022 | Class điều kiện, 1024², ImageNet. |
| R3GAN | 2024 | Đổi thương hiệu với reg mạnh hơn; thu hẹp khoảng cách khuếch tán trên FFHQ-1024 với ít tham số hơn 20 lần. |

Vào năm 2026, StyleGAN3 vẫn là mặc định cho (a) chủ nghĩa ảnh chân thực miền hẹp ở FPS cao, (b) few-shot thích ứng miền (huấn luyện trên một dataset mới với 100 hình ảnh, lập bản đồ đóng băng), (c) chỉnh sửa dựa trên đảo ngược (tìm `w` tái tạo lại ảnh thực, sau đó chỉnh sửa `w` đó). Đối với chuyển văn bản thành hình ảnh miền mở, nó không phải là công cụ - khuếch tán là công cụ.

## Tự xây dựng

`code/main.py` triển khai đồ chơi "style-GAN lite" trong 1-D: MLP ánh xạ, một hàm tổng hợp lấy một vector hằng số đã học và điều chỉnh nó với scale/bias có nguồn gốc từ `w` và nhiễu trên mỗi lớp. Nó cho thấy rằng việc đưa `w` thông qua điều chế affin khớp hoặc nhịp nối `z` vào đầu vào của máy phát điện.

### Bước 1: lập bản đồ mạng

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### Bước 2: Chuẩn hóa phiên bản thích ứng

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

Tỷ lệ và bias trên mỗi feature bản đồ đến từ `w` thông qua phép chiếu tuyến tính.

### Bước 3: nhiễu trên mỗi lớp

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

Sigma trên mỗi kênh có thể học được.

## Cạm bẫy

- **Droplet artifacts.** StyleGAN 1 đã tạo ra một giọt blobby trong bản đồ feature vì AdaIN đã loại bỏ giá trị trung bình. Giải điều chế trọng lượng của StyleGAN 2 khắc phục nó bằng cách thay vào đó chia tỷ lệ trọng lượng tích chập.
- **Kết cấu dính.** Kết cấu StyleGAN 1 và 2 tuân theo tọa độ pixel, không phải tọa độ đối tượng (hiển thị khi nội suy). Các tích chập không có bí danh của StyleGAN 3 khắc phục điều này bằng các bộ lọc sinc có cửa sổ.
- **Phạm vi chế độ.** `ψ < 0.7` cắt bớt trông sạch sẽ nhưng samples từ một hình nón hẹp; Sử dụng `ψ = 1.0` nếu bạn cần sự đa dạng.
- **Đảo ngược là mất dữ liệu.** Đảo ngược ảnh thực thành `W` thường được thực hiện thông qua tối ưu hóa hoặc encoder (e4e, ReStyle, HyperStyle). Kết quả trôi qua nhiều lần lặp lại.

## Ứng dụng

| Trường hợp sử dụng | Cách tiếp cận |
|----------|----------|
| Khuôn mặt người chân thực (anime, sản phẩm, hẹp) | StyleGAN3 FFHQ / fine-tune tùy chỉnh |
| Chỉnh sửa khuôn mặt từ ảnh | đảo ngược e4e + hướng StyleSpace / InterFaceGAN |
| Hoán đổi khuôn mặt / tái hiện | StyleGAN + encoder + pha trộn |
| Hình đại diện pipelines | StyleGAN3 w / ADA cho fine-tune dữ liệu thấp |
| Thích ứng miền từ một vài hình ảnh | Đóng băng ánh xứ, fine-tune tổng hợp |
| Tạo đa phương thức hoặc có điều kiện văn bản | Đừng - sử dụng khuếch tán |

Đối với các bản demo cấp sản phẩm mà câu trả lời là "ảnh khuôn mặt của một người", StyleGAN đánh bại sự khuếch tán về chi phí inference (forward pass đơn, <10ms trên 4090) và độ sắc nét cho cùng một thanh chất lượng.

## Sản phẩm bàn giao

Tiết kiệm `outputs/skill-stylegan-inversion.md`. Skill chụp ảnh thực và kết quả: phương pháp đảo ngược (e4e / ReStyle / HyperStyle), loss tiềm ẩn dự kiến, ngân sách chỉnh sửa (bạn có thể di chuyển bao xa trong `W` trước artifacts) và danh sách các hướng chỉnh sửa tốt đã biết (tuổi, biểu cảm, tư thế).

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py` với `adain_on=True` và `adain_on=False`. So sánh sự lan truyền của các đầu ra cho một tiềm ẩn cố định và tiềm ẩn bị nhiễu loạn.
2. **Trung bình.** Thực hiện chính quy hóa trộn: đối với một training batch, tính toán `w_a`, `w_b` và áp dụng `w_a` cho nửa đầu của quá trình tổng hợp và `w_b` cho nửa sau. Các decoder có học được các phong cách được gỡ rối không?
3. **Khó.** Lấy pretrained StyleGAN3 FFHQ model (ffhq-1024.pkl). Tìm hướng `w` điều khiển "nụ cười" bằng cách training SVM trên các mẫu được dán nhãn; Báo cáo bạn có thể đẩy bao xa trước khi danh tính trôi dạt.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Mạng lập bản đồ | "MLP" | `f: Z → W`, 8 lớp, tách hình học tiềm ẩn khỏi thống kê dữ liệu. |
| Không gian W | "Không gian phong cách" | Đầu ra của mạng bản đồ; gần như không bị rối rắm. |
| AdaIN | "Định mức phiên bản thích ứng" | Chuẩn hóa bản đồ feature, sau đó chia tỷ lệ + dịch chuyển bằng phép chiếu `w`. |
| Thủ thuật cắt bớt | "Psi" | `w = mean + ψ·(w - mean)`, ψ<1 đánh đổi sự đa dạng cho chất lượng. |
| Chính quy hóa độ dài đường dẫn | "Đăng ký PL" | Phạt những thay đổi lớn về hình ảnh trên mỗi đơn vị thay đổi trong `w`; làm cho `W` mượt mà hơn. |
| Giải điều chế trọng lượng | "Bản sửa lỗi StyleGAN2" | Chuẩn hóa trọng số conv thay vì kích hoạt; tiêu diệt giọt artifacts. |
| Không có bí danh | "Thủ thuật của StyleGAN3" | Bộ lọc sinc có cửa sổ; Loại bỏ kết cấu dính vào lưới pixel. |
| Đảo ngược | "Tìm w cho một hình ảnh thực" | Tối ưu hóa hoặc mã hóa `x → w` để `G(w) ≈ x`. |

## Production lưu ý: tại sao StyleGAN vẫn ships vào năm 2026

StyleGAN3 trên 4090 tạo ra mặt FFHQ 1024² trong vòng chưa đầy 10 ms - `num_steps = 1`, không giải mã VAE, không có cross-attention vượt qua. Nói một cách production, đây là độ trễ sàn cho bất kỳ trình tạo hình ảnh nào. pipeline giải mã SDXL + VAE 50 bước ở cùng độ phân giải là ~3 giây. Đó là khoảng cách **300×** và đối với các sản phẩm tên miền hẹp (dịch vụ hình đại diện, pipelines giấy tờ tùy thân, tạo khuôn mặt chứng khoán), nó giành chiến thắng trên TCO.

Hai hậu quả hoạt động:

- **Không có bộ lập lịch, không có bộ phân đào.** batch tĩnh ở mức sử dụng mục tiêu là tối ưu. Batching liên tục (cần thiết cho LLMs và khuếch tán) không mang lại lợi ích gì vì mọi yêu cầu đều có cùng một FLOPs.
- **`ψ` cắt ngắn là núm an toàn.** `ψ < 0.7` samples từ một hình nón hẹp trong phạm vi của mạng ánh xạ. Đây là đòn bẩy duy nhất mà lớp phục vụ có trên variance mẫu. Giảm `ψ` ở mức tải cao nhất, nâng nó lên cho người dùng cao cấp.

## Đọc thêm

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) - StyleGAN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) - StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) - StyleGAN3.
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) - Đảo ngược E4E.
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) - StyleGAN-XL.
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) - công thức GAN tối giản hiện đại.
