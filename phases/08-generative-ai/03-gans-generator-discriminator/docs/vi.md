# GAN - Máy phát điện vs Bộ phân biệt đối xử

> Thủ thuật của Goodfellow vào năm 2014 là bỏ qua hoàn toàn mật độ. Hai mạng. Một người làm giả. Một người bắt được chúng. Họ chiến đấu cho đến khi hàng giả không thể phân biệt được với thật. Nó không nên hoạt động. Nó thường không. Khi nó xảy ra, các mẫu vẫn sắc nét nhất trong tài liệu cho các miền hẹp.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 3 · 02 (Backprop), Giai đoạn 3 · 08 (Optimizers), Giai đoạn 8 · 02 (VAE)
**Thời lượng:** ~75 phút

## Vấn đề

VAE tạo ra các mẫu mờ vì decoder loss MSE của chúng là Bayes tối ưu cho hình ảnh * trung bình * - và giá trị trung bình của nhiều chữ số hợp lý là một chữ số mờ. Bạn muốn một loss thưởng *tính hợp lý*, chứ không phải khoảng cách theo pixel với bất kỳ mục tiêu nào. Không có hình thức đóng cho tính hợp lý. Bạn phải học nó.

Ý tưởng của Goodfellow: huấn luyện một bộ phân loại `D(x)` phân biệt hình ảnh thật với hình ảnh giả. Huấn luyện một máy phát điện `G(z)` để đánh lừa `D`. Tín hiệu loss cho `G` là bất cứ điều gì `D` hiện đang nghĩ làm cho một cái gì đó trông như thật. Tín hiệu này cập nhật khi `G` cải thiện, đuổi theo mục tiêu di chuyển. Nếu cả hai mạng hội tụ, `G` đã học được sự phân phối dữ liệu mà không bao giờ viết ra `log p(x)`.

Đây là training đối nghịch. Toán học là một trò chơi minimax:

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

Vào năm 2026, GAN không còn là máy tạo SOTA nữa (khuếch tán và dòng chảy phù hợp với vương miện đó). Nhưng StyleGAN 2/3 vẫn là khuôn mặt sắc nét nhất models từng shipped, bộ phân biệt GAN được sử dụng làm *tổn thất nhận thức* trong training khuếch tán và training đối nghịch cung cấp năng lượng cho quá trình chưng cất 1 bước nhanh (SDXL-Turbo, SD3-Turbo, LCM) cho phép bạn ship khuếch tán thời gian thực.

## Khái niệm

![GAN training: generator and discriminator in minimax](../assets/gan.svg)

**Máy phát điện `G(z)`.** Ánh xạ vector `z ~ N(0, I)` nhiễu đến `x̂` mẫu. Một mạng hình decoder (conv dày đặc hoặc chuyển vị).

**Discriminator `D(x)`.** Ánh xạ một mẫu với xác suất vô hướng (hoặc điểm). → thật 1, giả → 0.

**Loss.** Hai bản cập nhật xen kẽ:

- **Tàu `D`: **`loss_D = -[ log D(x) + log(1 - D(G(z))) ]`. Entropy chéo nhị phân trên thực = 1, giả = 0.
- **Tàu `G`: **`loss_G = -log D(G(z))`. Đây là dạng * không bão hòa * mà Goodfellow đã sử dụng (ban đầu `log(1 - D(G(z)))` bão hòa và giết chết gradients khi `D` tự tin).

**Training vòng lặp.** Một bước của `D`, một bước của `G`. Lặp lại.

**Tại sao nó hoạt động.** Nếu `G` hoàn toàn phù hợp với `p_data`, thì `D` không thể làm tốt hơn cơ hội và xuất ra 0,5 ở khắp mọi nơi; `G` không còn gradient nữa. Cân bằng.

**Tại sao nó bị hỏng.** Chế độ sụp đổ (`G` tìm thấy một chế độ `D` không thể phân loại và đúc nó mãi mãi), biến mất gradient (`D` học quá nhanh và `log D` bão hòa), training sự không ổn định (tốc độ học tập, kích thước batch, bất cứ thứ gì).

## Các biến thể làm cho GAN hoạt động

| Năm | Đổi mới | Sửa chữa |
|------|------------|-----|
| 2015 | DCGAN | Conv/deconv, batch tiêu chuẩn, LeakyReLU - kiến trúc ổn định đầu tiên. |
| 2017 | WGAN, WGAN-GP | Thay thế BCE bằng khoảng cách Wasserstein + gradient hình phạt. Sửa lỗi biến mất gradient. |
| 2017 | Chuẩn hóa quang phổ | Lipschitz ràng buộc kẻ phân biệt đối xử. Vẫn được sử dụng vào năm 2026 phân biệt đối xử. |
| 2018 | GAN lũy tiến | Huấn luyện độ phân giải thấp trước, thêm các lớp. Kết quả megapixel đầu tiên. |
| 2019 | Phong cáchGAN / Phong cách GAN2 | Ánh xạ mạng + định mức phiên bản thích ứng. Hiện đại cho chủ nghĩa ảnh chân thực miền cố định. |
| 2021 | Phong cáchGAN3 | Không có bí danh, phiên dịch tương đương - vẫn là tiêu chuẩn vàng của khuôn mặt vào năm 2026. |
| 2022 | Phong cáchGAN-XL | Có điều kiện, nhận thức class, quy mô lớn hơn. |
| 2024 | R3GAN | Đổi thương hiệu với chính quy hóa mạnh mẽ hơn; Hoạt động trên 1024² mà không cần thủ thuật. |

```figure
gan-minimax
```

## Tự xây dựng

`code/main.py` huấn luyện một GAN nhỏ trên dữ liệu 1-D: hỗn hợp của hai Gaussian. Máy phát điện và bộ phân biệt là các MLP một lớp ẩn. Chúng ta thực hiện vòng lặp tiến, lùi và minimax bằng tay. Mục tiêu là để xem hai chế độ lỗi chính (chế độ sụp đổ + gradient biến mất) khi chúng xảy ra.

### Bước 1: loss không bão hòa

Vanilla Goodfellow loss `log(1 - D(G(z)))` về 0 khi D phân loại hàng giả của G là giả với độ tin cậy cao. Tại thời điểm đó, gradient cho G về cơ bản là bằng không - G không thể cải thiện. Dạng không bão hòa `-log D(G(z))` có tiệm cận ngược lại: nó nổ tung khi D tự tin, tạo cho G một tín hiệu mạnh.

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### Bước 2: một bước phân biệt cho mỗi bước máy phát điện

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

Hàng giả mới cho G, nếu không gradients đã cũ.

### Bước 3: theo dõi thu gọn chế độ

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

Triệu chứng kinh điển: một trong hai chế độ thực ngừng được tạo ra. Người phân biệt đối xử ngừng sửa nó vì nó không bao giờ được coi là giả mạo.

## Cạm bẫy

- **Phân biệt quá mạnh.** Giảm learning rate của D 2-5x hoặc thêm instance/layer nhiễu. Nếu D đạt >95% accuracy, G đã chết.
- **Máy phát ghi nhớ một chế độ.** Thêm nhiễu vào đầu vào D, sử dụng lớp phân biệt minibatch hoặc chuyển sang WGAN-GP.
- **Batch thống kê rò rỉ định mức.** batch thật + batch giả chảy qua cùng một lớp BN trộn lẫn số liệu thống kê của chúng. Thay vào đó, hãy sử dụng định mức cá thể hoặc định mức quang phổ.
- **Trò chơi điểm khởi động.** FID và IS ồn ào ở số lượng mẫu thấp. Sử dụng ≥10k samples khi đánh giá.
- **One-shot sampling là một lời nói dối đối với các tác vụ có điều kiện.** Bạn vẫn cần thang đo CFG, thủ thuật cắt bớt và sampling lại để có được kết quả đầu ra có thể sử dụng được.

## Ứng dụng

stack GAN 2026:

| Tình huống | Chọn |
|-----------|------|
| Khuôn mặt người chân thực, tư thế cố định | StyleGAN3 (sắc nét nhất, nhỏ nhất) |
| Anime / khuôn mặt cách điệu | StyleGAN-XL hoặc LoRA khuếch tán ổn định |
| Dịch từ hình ảnh sang hình ảnh | Pix2Pix / CycleGAN (Giai đoạn 8 · 04) hoặc ControlNet (Giai đoạn 8 · 08) |
| Chuyển văn bản thành hình ảnh 1 bước nhanh chóng | distillation khuếch tán đối nghịch (SDXL-Turbo, SD3-Turbo) |
| loss nhận thức bên trong máy huấn luyện khuếch tán | Bộ phân biệt GAN nhỏ trên cắt hình ảnh |
| Bất cứ thứ gì đa phương thức, kết thúc mở | Không nên — sử dụng khuếch tán hoặc kết hợp dòng chảy |

GAN sắc nét nhưng hẹp. Khi miền của bạn mở ra — ảnh, prompts văn bản tùy ý, video — hãy chuyển sang phổ tán. Thủ thuật đối nghịch tồn tại như một thành phần (mất nhận thức, distillation), không phải là một máy phát độc lập.

## Sản phẩm bàn giao

Tiết kiệm `outputs/skill-gan-debugger.md`. Skill chạy GAN không thành công (đường cong loss, lưới mẫu, kích thước dataset) và xuất ra danh sách xếp hạng các nguyên nhân có thể xảy ra, bản sửa lỗi một dòng và giao thức chạy lại.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py` với cài đặt gốc. Sau đó đặt `D_LR = 5 * G_LR` và chạy lại. loss của G sụp đổ thành một hằng số nhanh như thế nào?
2. **Trung bình.** Thay thế loss Goodfellow BCE bằng loss WGAN: `loss_D = E[D(fake)] - E[D(real)]`, `loss_G = -E[D(fake)]` và kẹp trọng lượng của D thành `[-0.01, 0.01]`. training có ổn định hơn không? So sánh sự hội tụ của đồng hồ treo tường.
3. **Khó.** Mở rộng ví dụ 1-D thành dữ liệu 2-D (hỗn hợp 8 Gaussian trên một vòng). Theo dõi có bao nhiêu trong số 8 chế độ mà trình tạo chụp ở các bước 1k, 5k, 10k. Thực hiện phân biệt và đo lường lại lô nhỏ.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Máy phát điện | "G" | Mạng nhiễu thành mẫu, `G: z → x̂`. |
| Phân biệt đối xử | "D" | Bộ phân loại `D: x → [0, 1]`, thật và giả. |
| Tối đa | "Trò chơi" | `min_G max_D` của một mục tiêu chung. |
| loss không bão hòa | "Sửa chữa" | Sử dụng `-log D(G(z))` cho G thay vì `log(1 - D(G(z)))`. |
| Thu gọn chế độ | "G ghi nhớ một điều" | Generator tạo ra một số đầu ra riêng biệt mặc dù dữ liệu đa dạng. |
| WGAN | "Wasserstein" | Thay thế BCE bằng khoảng cách Earth-Mover + hình phạt gradient; gradient mượt mà hơn. |
| Định mức quang phổ | "Thủ thuật Lipschitz" | Hạn chế các định mức trọng lượng của D để ràng buộc độ dốc của nó; ổn định training. |
| Phong cáchGAN | "Cái hiệu quả" | Mạng lập bản đồ + AdaIN; tốt nhất trong class cho khuôn mặt, vẫn vào năm 2026. |

## Lưu ý Production: one-shot inference là lợi thế lâu dài của GAN

GAN không còn chiến thắng về chất lượng mẫu để tạo miền mở, nhưng chúng vẫn giành chiến thắng về chi phí inference. Trong từ vựng văn học production-inference, GAN có:

- **Không cần điền trước, không có giai đoạn giải mã.** Một `G(z)` forward pass duy nhất. TTFT ≈ tổng độ trễ.
- **Không có áp suất bộ nhớ đệm KV.** Trạng thái duy nhất là trọng lượng. Kích thước Batch được giới hạn bởi bộ nhớ kích hoạt, không phải bộ nhớ đệm.
- **Hàng loạt liên tục tầm thường.** Vì mọi yêu cầu đều có cùng một FLOPs cố định, nên batch tĩnh ở mức sử dụng mục tiêu của server thường là tối ưu. Không cần lịch trình trên chuyến bay.

Đây là lý do tại sao GAN distillation (SDXL-Turbo, SD3-Turbo, ADD, LCM) là kỹ thuật thống trị để chuyển văn bản thành hình ảnh nhanh vào năm 2026: nó thu gọn pipeline khuếch tán 20-50 bước thành 1-4 đường chuyền về phía trước kiểu GAN trong khi vẫn giữ được sự phân bố của cơ sở khuếch tán. loss đối thủ tồn tại như một núm training thời gian để biến máy phát điện chậm thành máy phát điện nhanh.

## Đọc thêm

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — bài báo GAN gốc.
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) - kiến trúc ổn định đầu tiên.
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) - WGAN.
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) - SN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) - StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) - StyleGAN3.
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) - SDXL-Turbo.
