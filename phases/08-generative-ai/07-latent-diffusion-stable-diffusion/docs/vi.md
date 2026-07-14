# Khuếch tán tiềm ẩn và khuếch tán ổn định

> Khuếch tán không gian pixel trên 512×512 hình ảnh là một tội ác chiến tranh tính toán. Rombach et al. (2022) nhận thấy rằng bạn không cần tất cả các kích thước 786k để tạo ra một hình ảnh - bạn cần đủ để nắm bắt cấu trúc ngữ nghĩa và một decoder riêng cho rest. Chạy khuếch tán bên trong không gian tiềm ẩn của VAE. Một ý tưởng đó là Khuếch tán ổn định.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 8 · 02 (VAE), Giai đoạn 8 · 06 (DDPM), Giai đoạn 7 · 09 (ViT)
**Thời lượng:** ~75 phút

## Vấn đề

Sự khuếch tán không gian pixel ở 512² có nghĩa là U-Net chạy trên tensors hình dạng `[B, 3, 512, 512]`. Mỗi bước sampling là ~100 GFLOPS cho U-Net 500 triệu tham số. Năm mươi bước là 5 TFLOPS cho mỗi hình ảnh. Huấn luyện trên một tỷ hình ảnh và hóa đơn tính toán là vô lý.

Hầu hết những FLOPs đó đều đẩy các chi tiết không quan trọng về mặt nhận thức qua mạng - kết cấu tần số cao mà một VAE mất dữ liệu có thể nén đi. Ý tưởng của Rombach: huấn luyện VAE một lần (*giai đoạn đầu tiên*), đóng băng nó và chạy khuếch tán hoàn toàn trong không gian tiềm ẩn 4 kênh 64×64 (*giai đoạn thứ hai*). Cùng một U-Net. 1/16th các pixel. FLOPs ít hơn ~64 lần cho chất lượng tương đương.

Đây là công thức khuếch tán ổn định. SD 1.x / 2.x sử dụng U-Net 860M trên `64×64×4` tiềm ẩn, SDXL sử dụng U-Net 2.6B trên `128×128×4`, SD3 hoán đổi U-Net cho Transformer khuếch tán (DiT) với kết hợp dòng chảy. Flux.1-dev (Phòng thí nghiệm Rừng Đen, 2024) ships DiT-MMDiT 12B-param. Tất cả đều chạy trên cùng một chất nền hai giai đoạn.

## Khái niệm

![Latent diffusion: VAE compression + diffusion in latent space](../assets/latent-diffusion.svg)

**Hai giai đoạn, được huấn luyện riêng biệt.**

1. **Giai đoạn 1 - VAE.** Encoder `E(x) → z`, decoder `D(z) → x`. Nén mục tiêu: 8× downsample trong mỗi trục không gian + điều chỉnh các kênh để tổng kích thước tiềm ẩn là ~1/16th số điểm ảnh. Loss = tái tạo (L1 + LPIPS tri giác) + KL (trọng lượng nhỏ nên `z` không bị ép buộc quá Gaussian, vì chúng ta không cần sampling chính xác từ `z`). Thường được huấn luyện với một loss đối thủ nên hình ảnh được giải mã rất sắc nét.

2. **Giai đoạn 2 - khuếch tán trên `z`.** Coi `z = E(x_real)` là dữ liệu. Huấn luyện U-Net (hoặc DiT) để khử nhiễu `z_t`. Tại inference: lấy mẫu `z_0` thông qua khuếch tán, sau đó `x = D(z_0)`.

**Text conditioning.** Hai thành phần bổ sung. Một encoder văn bản bị đóng băng (CLIP-L cho SD 1.x, CLIP-L + OpenCLIP-G cho SD 2/XL, T5-XXL cho SD3 và Flux). Một mũi tiêm cross-attention: mỗi khối U-Net lấy `[Q = image features, K = V = text tokens]` và trộn chúng vào. tokens là cách duy nhất văn bản ảnh hưởng đến hình ảnh.

**Chức năng loss giống với Bài 06.** Cùng DDPM / luồng phù hợp với MSE về nhiễu. Bạn chỉ cần hoán đổi miền dữ liệu.

## Các biến thể kiến trúc

| Model | Năm | Xương sống | Hình dạng tiềm ẩn | encoder văn bản | Tham số |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 · | 2022 | Mạng U-Net | 64×64×4 | CLIP-L (77 tokens) | 860 triệu |
| SD 2.1 | 2022 | Mạng U-Net | 64×64×4 | Mở CLIP-H | 865 triệu |
| SDXL | 2023 | U-Net + máy lọc dầu | 128×128×4 · | CLIP-L + Mở CLIP-G | 2,6 tỷ + 6,6 tỷ |
| SDXL-Turbo | 2023 | Chưng cất | 128×128×4 · | giống nhau | 1-4 bước sampling |
| SD3 | 2024 | MMDiT (DiT đa phương thức) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-phát triển | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12 tỷ |
| Thông lượng.1-schnell | 2024 | MMDiT chưng cất | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 bước |

Xu hướng: thay thế U-Net bằng DiT (transformer trên các bản vá tiềm ẩn), mở rộng encoder văn bản (T5 đánh bại CLIP để tuân thủ prompt), tăng các kênh tiềm ẩn (4 → 16 cho khoảng trống chi tiết hơn).

```figure
noise-schedule
```

## Tự xây dựng

`code/main.py` stacks một đồ chơi 1-D "VAE" (nhận dạng encoder + decoder, để minh họa; một VAE thực sự sẽ là một mạng conv) trên DDPM từ Bài 06 và thêm điều kiện class với classifier-free guidance. Nó cho thấy rằng cùng một loss khuếch tán hoạt động cho dù bạn chạy trên các giá trị 1-D thô hay trên các giá trị được mã hóa - thông tin chi tiết chính.

### Bước 1: encoder/decoder

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

Một VAE thực sự đã tập tạ. Đối với sư phạm, bản đồ tuyến tính này đủ để cho thấy rằng sự khuếch tán hoạt động trên `z` mà không quan tâm đến không gian dữ liệu gốc.

### Bước 2: khuếch tán trong không gian `z`

DDPM tương tự như bài 06. Dữ liệu mà mạng nhìn thấy là `z = E(x)`. Sau khi sampling `z_0`, giải mã bằng `D(z_0)`.

### Bước 3: classifier-free guidance

Trong khi training, thả nhãn class 10% thời gian (thay thế bằng token rỗng). Tại inference, hãy tính cả `ε_cond` và `ε_uncond`, thì:

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = không có hướng dẫn (đa dạng đầy đủ), `w = 3` = mặc định, `w = 7+` = bão hòa / quá sắc nét.

### Bước 4: text conditioning (khái niệm, không phải mã)

Thay thế nhãn class bằng đầu ra encoder văn bản bị đóng băng. Cung cấp embedding văn bản đến U-Net qua cross-attention:

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

Đây là sự khác biệt đáng kể duy nhất giữa model khuếch tán class điều kiện và Khuếch tán ổn định.

## Cạm bẫy

- **Thang đo VAE không khớp.** SD 1.x VAE có hằng số tỷ lệ (`scaling_factor ≈ 0.18215`) được áp dụng sau khi mã hóa. Quên điều này làm cho U-Net huấn luyện trên các tiềm ẩn với variance cực kỳ sai lầm. Mỗi checkpoint ships một.
- **Văn bản encoder âm thầm sai.** SD3 cần T5-XXL với > = 128 tokens và dự phòng chỉ CLIP là mất dữ liệu. Luôn kiểm tra `use_t5=True` hoặc prompt miệng núi lửa có độ trung thực.
- **Trộn các không gian tiềm ẩn.** SDXL, SD3, Flux đều sử dụng các VAE khác nhau. Một LoRA được huấn luyện về SDXL tiềm ẩn sẽ không hoạt động trên SD3. Bộ khuếch tán Hugging Face 0.30+ từ chối tải checkpoints không khớp.
- **CFG quá cao.** `w > 10` tạo ra hình ảnh bão hòa, nhờn và quá phù hợp với prompt với cái giá phải trả là sự đa dạng. Điểm ngọt ngào là `w = 3-7`.
- **Âm prompts rò rỉ.** prompt âm rỗng trở thành token rỗng; một prompt âm được lấp đầy trở thành `ε_uncond`. Những điều này không giống nhau; một số pipelines âm thầm mặc định là rỗng.

## Ứng dụng

Production stacks vào năm 2026:

| Mục tiêu | Xương sống được đề xuất |
|--------|----------------------|
| Miền hẹp, dữ liệu được ghép nối training model từ đầu | SDXL fine-tune (LoRA / đầy đủ) — nhanh nhất để ship |
| Chuyển văn bản thành hình ảnh miền mở, trọng lượng mở | Flux.1-dev (12B, Apache / phi thương mại) hoặc SD3.5-Large |
| Tạ mở inference nhanh nhất | Flux.1-schnell (1-4 bước, Apache) hoặc SDXL-Lightning |
| Tuân thủ prompt tốt nhất, được lưu trữ | GPT-Image / DALL-E 3 (tĩnh), Midjourney v7, Imagen 4 |
| Chỉnh sửa quy trình làm việc | Flux.1-Kontext (Tháng 12 năm 2024) — chấp nhận hình ảnh + văn bản nguyên bản |
| Nghiên cứu, cơ sở | SD 1.5 — cổ xưa nhưng được nghiên cứu kỹ lưỡng |

## Sản phẩm bàn giao

Tiết kiệm `outputs/skill-sd-prompter.md`. Skill lấy prompt văn bản + kiểu mục tiêu và đầu ra: model + checkpoint, thang đo CFG, bộ lấy mẫu, prompt âm, độ phân giải, kết hợp ControlNet/IP-Adapter tùy chọn và danh sách kiểm tra QA theo bước.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py` với `w ∈ {0, 1, 3, 7, 15}` hướng dẫn. Ghi mẫu trung bình bằng class. Phương tiện class phân kỳ qua các phương tiện dữ liệu thực ở `w` nào?
2. **Trung bình.** Đổi encoder tuyến tính đồ chơi lấy cặp encoder/decoder tanh-MLP với loss tái tạo. Huấn luyện lại sự khuếch tán trên các tiềm ẩn mới. Chất lượng mẫu có thay đổi không?
3. **Khó.** Thiết lập một inference khuếch tán ổn định thực sự với bộ khuếch tán: tải `sdxl-base`, chạy 30 bước Euler với CFG = 7, tính thời gian. Bây giờ chuyển sang `sdxl-turbo` với 4 bước và CFG = 0. Cùng một chủ đề, chất lượng khác nhau - mô tả những gì đã thay đổi và tại sao.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Giai đoạn đầu tiên | "VAE" | Cặp encoder/decoder được huấn luyện; nén 512² thành 64². |
| Giai đoạn thứ hai | "Mạng chữ U" | Sự khuếch tán model trên không gian tiềm ẩn. |
| CFG | "Thang điểm hướng dẫn" | `(1+w)·ε_cond - w·ε_uncond`; điều chỉnh sức mạnh điều hòa. |
| token rỗng | "Nhúng prompt trống" | Nhúng vô điều kiện được sử dụng cho `ε_uncond`. |
| Cross-attention | "Làm thế nào văn bản đi vào" | Mỗi khối U-Net tham gia vào các tokens văn bản dưới dạng K và V. |
| DiT | "Khuếch tán Transformer" | Thay thế U-Net bằng một transformer trên các bản vá tiềm ẩn; quy mô tốt hơn. |
| MMDiT | "DiT đa phương thức" | Kiến trúc của SD3: luồng văn bản và hình ảnh với attention chung. |
| Hệ số tỷ lệ VAE | "Số ma thuật" | Chia các tiềm ẩn cho ~5,4 để khuếch tán hoạt động trong không gian đơn vị variance. |

## Lưu ý Production: chạy Flux-12B trên GPU tiêu dùng 8GB

tích hợp Flux tham khảo là công thức chuẩn "Tôi có GPU tiêu dùng, tôi có thể ship cái này không?". Bí quyết là cùng một công thức ba núm production inference danh sách tài liệu được áp dụng cho DiT khuếch tán:

1. **Tải so le.** Flux có ba mạng không bao giờ cần cùng tồn tại trong VRAM: encoder văn bản T5-XXL (~10 GB trong fp32), CLIP-L (nhỏ), MMDiT 12B và VAE. Mã hóa prompt trước, * xóa * encoders, tải DiT, khử nhiễu, * xóa * DiT, tải VAE, giải mã. Người tiêu dùng 8GB GPUs chỉ phù hợp với một giai đoạn tại một thời điểm.
2. **quantization 4 bit qua bitsandbytes.** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)` trên cả encoder T5 và DiT. Giảm bộ nhớ 8×, chất lượng giảm không thể nhận thấy đối với chuyển văn bản thành hình ảnh theo benchmarks của Aritra (được liên kết trong sổ tay).
3. **CPU giảm tải.** `pipe.enable_model_cpu_offload()` tự động hoán đổi các mô-đun giữa CPU và GPU khi mỗi forward pass tiến triển. Thêm độ trễ 10-20% nhưng làm cho pipeline chạy.

Tính toán bộ nhớ là: `10 GB T5 / 8 = 1.25 GB` lượng tử hóa, `12 B params × 0.5 bytes = ~6 GB` lượng tử hóa DiT, cộng với các kích hoạt. Theo thuật ngữ của stas00, đây là cực điểm của TP = 1 inference - không model song song, quantization tối đa. Đối với production, bạn sẽ chạy TP = 2 hoặc TP = 4 trên H100; Đối với một máy tính xách tay dành cho nhà phát triển, đây là công thức.

## Đọc thêm

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) - Khuếch tán ổn định.
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) - SDXL.
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) - DiT.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) - SD3, MMDiT.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) - CFG.
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) - Gia đình Flux.1.
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — triển khai tham chiếu cho mọi checkpoint ở trên.
