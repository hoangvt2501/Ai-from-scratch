# Bộ mã hóa tự động và bộ mã hóa tự động biến thể (VAE)

> Một bộ mã hóa tự động đơn giản nén sau đó tái tạo. Nó ghi nhớ. Nó không tạo ra. Thêm một thủ thuật - buộc mã trông giống Gaussian - và bạn sẽ có một bộ lấy mẫu. Thủ thuật duy nhất đó, tham số hóa lại `z = μ + σ·ε`, là lý do tại sao mọi model hình ảnh khuếch tán tiềm ẩn và khớp luồng mà bạn sử dụng vào năm 2026 đều có VAE ở đầu vào.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 3 · 02 (Backprop), Giai đoạn 3 · 07 (CNN), Giai đoạn 8 · 01 (Phân loại)
**Thời lượng:** ~75 phút

## Vấn đề

Nén chữ số MNIST 784 pixel thành mã 16 số, sau đó tạo lại. Một bộ mã hóa tự động đơn giản sẽ vượt qua MSE tái tạo nhưng không gian mã là một mớ hỗn độn. Chọn một điểm ngẫu nhiên trong không gian mã, giải mã nó và bạn sẽ nhận được nhiễu. Nó không có bộ lấy mẫu. Đó là một nén model ăn mặc.

Những gì bạn thực sự muốn là: (a) không gian mã là một phân phối sạch, mượt mà mà bạn có thể lấy mẫu - giả sử một `N(0, I)` Gaussian đẳng hướng, (b) giải mã bất kỳ mẫu nào tạo ra một chữ số hợp lý, và (c) encoder và decoder vẫn nén tốt. Ba mục tiêu, một kiến trúc, một loss.

VAE năm 2013 của Kingma giải quyết vấn đề này bằng cách training encoder xuất ra `q(z|x) = N(μ(x), σ(x)²)` *phân phối*, kéo phân phối đó về phía prior `N(0, I)` thông qua hình phạt KL và sau đó sampling `z` từ `q(z|x)` trước khi giải mã. Tại thời điểm inference, thả encoder, lấy mẫu `z ~ N(0, I)`, giải mã. Hình phạt KL là thứ buộc không gian mã phải được cấu trúc.

Vào năm 2026, VAE hiếm khi ship độc lập - chúng đã vượt trội về chất lượng hình ảnh thô - nhưng chúng là encoder được lựa chọn cho mọi model khuếch tán tiềm ẩn (SD 1/2/XL/3, Flux, AudioCraft). Tìm hiểu VAE và bạn học lớp đầu tiên vô hình của mọi hình ảnh pipeline bạn sử dụng.

## Khái niệm

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**Autoencoder.** `z = encoder(x)`, `x̂ = decoder(z)`, loss = `||x - x̂||²`. Không gian mã không có cấu trúc.

**VAE encoder.** Xuất ra hai vectors: `μ(x)` và `log σ²(x)`. Những điều này xác định `q(z|x) = N(μ, diag(σ²))`.

**Thủ thuật tái tham số.** Sampling từ `q(z|x)` không thể phân biệt được. Viết lại mẫu như `z = μ + σ·ε` nơi `ε ~ N(0, I)`. Bây giờ `z` là một hàm xác định của `(μ, σ)` cộng với nhiễu không parameter - gradients chảy qua `μ` và `σ`.

**Loss.** Bằng chứng BOund thấp hơn (ELBO), hai thuật ngữ:

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

Tái thiết đẩy `x̂` hướng tới `x`. KL đẩy `q(z|x)` về phía prior. Họ đánh đổi. β nhỏ (<1) = mẫu sắc nét hơn, không gian mã ít Gaussian hơn. β lớn (>1) = không gian mã sạch hơn, mẫu mờ hơn. β-VAE (Higgins 2017) đã làm cho núm này trở nên nổi tiếng và bắt đầu nghiên cứu gỡ rối.

**Sampling.** Tại inference: vẽ `z ~ N(0, I)`, chuyển tiếp qua decoder. Một forward pass - không có sampling lặp đi lặp lại như khuếch tán.

```figure
vae-latent-grid
```

## Tự xây dựng

`code/main.py` triển khai một VAE nhỏ mà không cần numpy hoặc ngọn đuốc. Đầu vào là dữ liệu tổng hợp 8 chiều được lấy từ hỗn hợp Gaussian 2 thành phần ở dạng 8-D. Encoder và decoder là các MLP lớp ẩn đơn. Chúng ta triển khai kích hoạt tảnh, forward pass, loss và backward pass viết tay. Không phải production - sư phạm.

### Bước 1: encoder về phía trước

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

`log σ²` thay vì `σ` nên đầu ra mạng không bị hạn chế (softplus của σ là một cái bẫy - gradients chết ở σ ≈ 0).

### Bước 2: tham số lại và giải mã

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### Bước 3: ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

KL dạng đóng chính xác vì cả hai phân phối đều là Gaussian. Không tích hợp bằng số. Mọi người vẫn ship mã với ước tính của monte-carlo KL vào năm 2026 - nó chậm hơn gấp 3 lần mà không có lý do.

### Bước 4: tạo

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

Đó là model sinh sản. Năm dòng.

## Cạm bẫy

- **Posterior sụp đổ.** Thuật ngữ KL thúc đẩy `q(z|x) → N(0, I)` mạnh mẽ đến mức `z` không mang thông tin về `x`. Khắc phục: ủ β (bắt đầu β = 0, ramp đến 1), bit tự do hoặc bỏ qua KL trên các kích thước không hoạt động.
- **Mẫu mờ.** decoder likelihood Gaussian ngụ ý tái tạo MSE, là Bayes tối ưu cho L2 (giá trị trung bình) - giá trị trung bình của một tập hợp các chữ số hợp lý là một chữ số mờ. Khắc phục: decoder rời rạc (VQ-VAE, NVAE), hoặc chỉ sử dụng VAE như một encoder và khuếch tán stack trên tiềm ẩn (đây là những gì Khuếch tán ổn định làm).
- **β quá lớn, quá sớm.** Xem posterior sụp đổ. Bắt đầu từ β≈0.01 và đường dốc.
- **Độ mờ tiềm ẩn quá nhỏ.** 16-D hoạt động cho MNIST, 256-D cho ImageNet 256², 2048-D cho ImageNet 1024². VAE của Stable Diffusion nén 512×512×3 → 64×64×4 (hệ số mẫu xuống 32x trong diện tích không gian, 32x trong kênh).

## Ứng dụng

VAE stack 2026:

| Tình huống | Chọn |
|-----------|------|
| encoder tiềm ẩn hình ảnh để khuếch tán | Khuếch tán ổn định VAE (`sd-vae-ft-ema`) hoặc Flux VAE |
| encoder tiềm ẩn âm thanh | Bộ mã hóa (Meta), SoundStream hoặc DAC (Mô tả) |
| Tiềm ẩn video | Các bản vá không gian của Sora, Latte VAE, WAN VAE |
| Học đại diện gỡ rối | β-VAE, Hệ sốVAE, TCVAE |
| Tiềm ẩn rời rạc (để tạo mô hình transformer) | VQ-VAE, RVQ (Vq dư) |
| Tiềm ẩn liên tục để tạo | VAE đơn giản, sau đó điều kiện một flow/diffusion model trong không gian tiềm ẩn đó |

model khuếch tán tiềm ẩn là VAE có model khuếch tán sống giữa encoder và decoder. VAE thực hiện nén thô, khuếch tán model nâng vật nặng. Cùng một mẫu cho video (VAE + DiT khuếch tán video) và âm thanh (Encodec + MusicGen transformer).

## Sản phẩm bàn giao

Tiết kiệm `outputs/skill-vae-trainer.md`.

Skill lấy: dataset hồ sơ + mục tiêu tiềm ẩn-mờ + sử dụng xuôi dòng (tái tạo, sampling hoặc đầu vào khuếch tán tiềm ẩn) và đầu ra: lựa chọn kiến trúc (trơn / β / VQ/RVQ), lịch trình β, độ mờ tiềm ẩn, decoder likelihood (Gaussian so với phân loại) và kế hoạch đánh giá (trinh sát MSE, KL mỗi độ mờ, khoảng cách Fréchet giữa `q(z|x)` và `N(0, I)`).

## Bài tập

1. **Dễ dàng.** Thay đổi `β` trong `code/main.py` thành `0.01`, `0.1`, `1.0`, `5.0`. Ghi lại quá trình tái tạo cuối cùng MSE và KL. Pareto-β nào tốt nhất cho dữ liệu tổng hợp của bạn?
2. **Trung bình.** Thay thế decoder likelihood Gaussian bằng likelihood Bernoulli (loss entropy chéo). So sánh chất lượng mẫu trên phiên bản nhị phân của cùng một dữ liệu tổng hợp.
3. **Khó.** Mở rộng `code/main.py` thành một VQ-VAE mini: thay thế `z` liên tục bằng tra cứu hàng xóm gần nhất trong sổ mã gồm các mục nhập K = 32. So sánh MSE tái tạo và báo cáo có bao nhiêu mục nhập sách mã được sử dụng (thu gọn sách mã là có thật).

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Bộ mã hóa tự động | Mạng mã hóa-giải mã | `x → z → x̂`, hãy học MSE. Không tạo ra. |
| VAE | AE với bộ lấy mẫu | Encoder xuất ra một bản phân phối, hình phạt KL định hình không gian mã. |
| ELBO | Bằng chứng giới hạn dưới | 'log p(x) ≥ trinh sát - KL[q(z\ | x) \ | \ | p(z)]`; tight when `q = p(z\ | x)`. |
| Tái tham số hóa | `z = μ + σ·ε` | Viết lại nút ngẫu nhiên dưới dạng xác định + nhiễu thuần túy. Cho phép backprop thông qua sampling. |
| Prior | `p(z)` | Phân phối mục tiêu cho tiềm ẩn, thường là `N(0, I)`. |
| Posterior sụp đổ | "Thuật ngữ KL chiến thắng" | Encoder bỏ qua `x`, xuất ra prior; decoder phải bị ảo giác. |
| β-VAE | Trọng lượng KL có thể điều chỉnh | `loss = recon + β·KL`. β cao hơn = gỡ rối hơn nhưng mờ hơn. |
| VQ-VAE | Tiềm ẩn rời rạc | Thay thế `z` liên tục bằng vector sách mã gần nhất; cho phép mô hình transformer. |

## Production lưu ý: VAE là đường dẫn nóng nhất trong server khuếch tán

Trong pipeline khuếch tán / thông lượng / SD3 ổn định, VAE được gọi hai lần cho mỗi yêu cầu - một lần để mã hóa (nếu thực hiện img2img / inpainting) và một lần để giải mã. Ở 1024², decoder vượt qua thường là đỉnh bộ nhớ kích hoạt lớn nhất trong toàn bộ pipeline vì nó nâng các mẫu `128×128×16` tiềm ẩn trở lại `1024×1024×3`. Hai hậu quả thực tế:

- **Cắt hoặc xếp lớp giải mã.** `diffusers` hiển thị `pipe.vae.enable_slicing()` và `pipe.vae.enable_tiling()`. Lát gạch đánh đổi một đường nối nhỏ artifact cho bộ nhớ `O(tile²)` thay vì `O(H·W)`. Cần thiết cho 1024²+ trên GPUs tiêu dùng.
- **bf16 decoder, số fp32 để thay đổi kích thước cuối cùng.** SD 1.x VAE được phát hành ở fp32 và *âm thầm tạo ra NaN* khi truyền sang fp16 ở 1024²+. SDXL ships `madebyollin/sdxl-vae-fp16-fix` - luôn thích biến thể fp16-fix hoặc sử dụng bf16.

## Đọc thêm

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - bài báo VAE.
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) - gỡ rối β-VAE.
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE.
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — hình ảnh hiện đại VAE.
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) - Khuếch tán ổn định; VAE như encoder.
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) - Encodec, tiêu chuẩn VAE âm thanh.
