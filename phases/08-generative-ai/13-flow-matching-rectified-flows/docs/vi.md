# Kết hợp dòng chảy và dòng điều chỉnh

> Khuếch tán models mất 20-50 bước sampling vì chúng đi theo một con đường cong từ nhiễu đến dữ liệu. Kết hợp dòng chảy (Lipman và cộng sự, 2023) và dòng chảy chỉnh lưu (Liu và cộng sự, 2022) đã huấn luyện các đường dẫn thẳng. Đường dẫn thẳng hơn có nghĩa là ít bước hơn có nghĩa là inference nhanh hơn. Stable Diffusion 3, Flux.1 và AudioCraft 2 đều chuyển sang kết hợp luồng vào năm 2024.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 8 · 06 (DDPM), Giai đoạn 1 · Giải tích
**Thời lượng:** ~45 phút

## Vấn đề

process ngược của DDPM là một bước đi ngẫu nhiên 1000 bước từ `N(0, I)` trở lại phân phối dữ liệu. DDIM đã thu gọn nó xuống còn 20-50 bước xác định. Bạn muốn ít bước hơn - lý tưởng nhất là một. Yếu tố cản trở là ODE giải process ngược lại là cứng; con đường cong.

Nếu bạn có thể huấn luyện model sao cho đường dẫn từ nhiễu đến dữ liệu là một *đường thẳng*, một bước Euler duy nhất từ `t=1` đến `t=0` sẽ hoạt động. Flow matching xây dựng điều này trực tiếp: xác định nội suy đường thẳng từ `x_1 ∼ N(0, I)` đến `x_0 ∼ data`, huấn luyện trường vector `v_θ(x, t)` để khớp đạo hàm thời gian của nó, tích hợp tại inference.

Dòng chỉnh lưu (Liu 2022) đi xa hơn: lặp đi lặp lại các đường dẫn bằng quy trình dàn lại tạo ra ODE tuyến tính gần hơn dần dần. Sau hai lần lặp lại nóng chảy, bộ lấy mẫu 2 bước phù hợp với chất lượng DDPM 50 bước.

## Khái niệm

![Flow matching: straight-line interpolation between noise and data](../assets/flow-matching.svg)

### Dòng chảy đường thẳng

Xác định:

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

nơi `x_0 ~ data` và `x_1 ~ N(0, I)`. Đạo hàm thời gian dọc theo đường thẳng này là hằng số:

```
dx_t / dt = x_1 - x_0
```

Xác định một trường vector thần kinh `v_θ(x_t, t)` và huấn luyện nó để khớp với đạo hàm này:

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

Đây là **khớp luồng có điều kiện** loss (Lipman 2023). Training không có mô phỏng: bạn không bao giờ mở ODE. Chỉ cần lấy mẫu `(x_0, x_1, t)` và thoái lui.

### Sampling

Tại inference, tích hợp lĩnh vực vector đã học * ngược * theo thời gian:

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

Bắt đầu từ `x_1 ~ N(0, I)`, Euler-bước xuống `t=0`.

### Lưu lượng chỉnh lưu (Liu 2022)

Dòng chảy đường thẳng hoạt động nhưng các đường dẫn đã học *không thực sự thẳng* - chúng cong vì nhiều `x_0` có thể ánh xạ đến cùng một `x_1`. Bước sắp xếp lại của luồng chỉnh lưu:

1. Dòng chảy tàu model v_1 với các cặp ngẫu nhiên.
2. Các cặp mẫu N `(x_1, x_0)` bằng cách tích hợp v_1 từ `x_1` đến `x_0` hạ cánh của nó.
3. Huấn luyện v_2 về các ví dụ được ghép đôi đó. Bởi vì các cặp bây giờ đã được "khớp với ODE", nội suy đường thẳng giữa chúng thực sự phẳng hơn.
4. Lặp lại.

Trong thực tế, 2 lần lặp lại sẽ giúp bạn gần tuyến tính, cho phép 2-4 bước inference. SDXL-Turbo, SD3-Turbo, LCM đều là models chưng cất từ dòng chảy.

### Tại sao điều này giành chiến thắng cho hình ảnh vào năm 2024

Ba lý do:

1. **training không mô phỏng **- không có ODE mở ra trong quá trình training, tầm thường để triển khai.
2. **Hình học loss tốt hơn** - đường dẫn thẳng có tín hiệu trên nhiễu nhất quán, trong khi DDPM ε-loss có SNR kém ở các cạnh của lịch trình.
3. **inference nhanh hơn **- 4-8 bước ở chất lượng SDXL-Turbo; 1 bước với distillation nhất quán.

## Kết hợp luồng so với DDPM — kết nối chính xác

Khớp dòng chảy với đường dẫn có điều kiện Gaussian là khuếch tán *với một lịch trình nhiễu cụ thể*. Chọn lịch trình `x_t = α(t) x_0 + σ(t) x_1` và kết hợp dòng chảy khôi phục sự khuếch tán được cải tiến bởi Stratonovich với `v = α'·x_0 - σ'·x_1`. Cả hai tương đương về mặt đại số cho các đường Gaussian.

Kết hợp dòng chảy nào được thêm vào: *độ rõ ràng* của mục tiêu (vận tốc đơn giản), loss sạch hơn và giấy phép thử nghiệm với các nội suy không phải Gaussian.

## Tự xây dựng

`code/main.py` thực hiện kết hợp dòng chảy 1-D trên hỗn hợp Gaussian hai chế độ. Trường vector `v_θ(x, t)` là một MLP nhỏ được huấn luyện với mục tiêu đường thẳng. Tại inference, tích hợp các bước 1, 2, 4 và 20 Euler và so sánh chất lượng mẫu.

### Bước 1: training loss

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### Bước 2: inference nhiều bước

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### Bước 3: so sánh số bước

Mong đợi bộ lấy mẫu 4 bước đã phù hợp với chất lượng 20 bước - một vấn đề lớn về độ trễ.

## Cạm bẫy

- **Tham số hóa thời gian.** Đối sánh luồng sử dụng `t ∈ [0, 1]` với `t=0` dữ liệu `t=1` nhiễu. DDPM sử dụng `t ∈ [0, T]` với `t=0` dữ liệu `t=T` nhiễu. Cùng một hướng, quy mô khác nhau. Giấy tờ liên tục làm sai điều này.
- **Lựa chọn lịch trình.** Đường thẳng của dòng chỉnh lưu là lịch trình phù hợp dòng chảy "the", nhưng bạn có thể sử dụng cosin hoặc t-sampling logit-normal (SD3 làm điều này) để bao phủ tỷ lệ tốt hơn.
- **Chi phí dàn lại.** Việc tạo dataset được ghép nối để dàn lại là một lần inference đầy đủ cho mỗi mẫu. Chỉ làm reflow khi bạn thực sự cần 1-2 bước inference.
- **Classifier-free guidance vẫn áp dụng.** Chỉ cần hoán đổi ε cho v trong tổ hợp tuyến tính: `v_cfg = (1+w) v_cond - w v_uncond`.

## Ứng dụng

| Trường hợp sử dụng | stack năm 2026 |
|----------|-----------|
| Chuyển văn bản thành hình ảnh, chất lượng tốt nhất | Kết hợp luồng: SD3, Flux.1-dev |
| Chuyển văn bản thành hình ảnh, 1-4 bước | Kết hợp dòng chưng cất: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| inference thời gian thực | Tính nhất quán distillation từ cơ sở phù hợp với dòng chảy (LCM, PCM) |
| Tạo âm thanh | Kết hợp luồng: Âm thanh ổn định 2.5, AudioCraft 2 |
| Tạo video | Kết hợp dòng chảy kết hợp với khuếch tán (Sora, Veo, Video ổn định) |
| Khoa học / vật lý (quỹ đạo hạt, phân tử) | Đối sánh luồng + trường vector tương đương |

Bất cứ khi nào một bài báo nói "nhanh hơn khuếch tán" vào năm 2025-2026, nó hầu như luôn là khớp dòng chảy + distillation.

## Sản phẩm bàn giao

Tiết kiệm `outputs/skill-fm-tuner.md`. Skill lấy thông số kỹ thuật model kiểu khuếch tán và chuyển đổi nó thành training config phù hợp với luồng: lựa chọn lịch trình, phân phối sampling thời gian (đồng nhất / logit-bình thường), optimizer, kế hoạch dàn lại, số bước mục tiêu, giao thức đánh giá.

## Bài tập

1. **Dễ dàng.** Chạy `code/main.py` và so sánh MSE 1 bước so với 20 bước với phân phối dữ liệu thực sự.
2. **Trung bình.** Chuyển từ `t` sampling đồng nhất sang logit bình thường (cô đặc sampling ở giữa t). Chất lượng model có được cải thiện không?
3. **Khó.** Thực hiện một lần lặp lại dàn lại: tạo cặp (x_0, x_1) bằng cách tích hợp model đầu tiên, huấn luyện model thứ hai trên các cặp và so sánh chất lượng mẫu 1 bước.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Kết hợp luồng | "Khuếch tán đường thẳng" | Huấn luyện `v_θ(x, t)` để khớp `x_1 - x_0` theo một nội suy. |
| Lưu lượng chỉnh lưu | "Dàn lại" | Quy trình lặp đi lặp lại làm thẳng các luồng đã học. |
| Trường vận tốc | "v_θ" | Đầu ra của model - hướng di chuyển `x_t`. |
| Nội suy đường thẳng | "Con đường" | `x_t = (1-t)·x_0 + t·x_1`; phái sinh mục tiêu tầm thường. |
| Bộ lấy mẫu Euler | "Bộ giải ODE bậc 1" | Nhà tích hợp đơn giản nhất; Hoạt động tốt khi đường dẫn thẳng. |
| Logit-bình thường t | "sampling SD3" | Tập trung `t` sampling hướng tới các giá trị trung bình, nơi gradients mạnh nhất. |
| Tính nhất quán distillation | "Bộ lấy mẫu 1 bước" | Huấn luyện học sinh ánh xạ bất kỳ `x_t` nào trực tiếp với `x_0`. |
| CFG với vận tốc | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`; cùng một thủ thuật, biến mới. |

## Lưu ý Production: Flux.1-schnell đang khớp luồng nhanh nhất

Chiến thắng production của Flow matching là Flux.1-schnell - một DiT phù hợp với dòng chảy được chắt lọc đến 1-4 bước inference trong khi vẫn giữ chất lượng cấp phát triển Flux. Máy tính xách tay "Run Flux trên máy 8GB" của Niels là công thức triển khai tham khảo: mã hóa T5 + CLIP, lượng tử hóa MMDiT denoise (trong 4 bước cho schnell so với 50 cho dev), giải mã VAE. Kế toán chi phí:

| Biến thể | Các bước | Độ trễ ở 1024² trên L4 | Tổng FLOPs (tương đối) |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (thô) | 50 | ~15 giây | 1.0× |
| Thông lượng.1-schnell | 4 | ~1,2 giây | 0,08× (nhanh hơn 12×) |
| Cơ sở SDXL | 30 | ~4 giây | 0,25× |
| SDXL-Lightning 2 bước | 2 | ~0,3 giây | 0,03× |

Quy tắc production: **cơ sở phù hợp với luồng + distillation = mặc định năm 2026 để chuyển văn bản thành hình ảnh nhanh.** Mọi nhà cung cấp lớn đều ships kết hợp này: SD3-Turbo (SD3 + lưu lượng + distillation), Flux-schnell (Flux-dev + làm thẳng dòng chỉnh lưu), CogView-4-Flash. Cơ sở khuếch tán thuần túy chỉ tồn tại cho các checkpoints kế thừa.

## Đọc thêm

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) - dòng chảy được chỉnh lưu.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) - kết hợp luồng.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) - SD3, lưu lượng chỉnh lưu trên quy mô lớn.
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) - framework chung bao gồm FM + khuếch tán.
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) - distillation 1 bước khuếch tán / dòng chảy.
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) - biến thể turbo.
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — kết hợp luồng trong production.
