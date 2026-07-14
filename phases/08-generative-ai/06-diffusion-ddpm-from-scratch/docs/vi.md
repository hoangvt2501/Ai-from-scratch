# Models khuếch tán — DDPM từ đầu

> Ho, Jain, Abbeel (2020) đã đưa ra lĩnh vực này một công thức mà nó không thể bỏ cuộc. Phá hủy dữ liệu với nhiễu qua hàng nghìn bước nhỏ. Huấn luyện một mạng nơ-ron để dự đoán nhiễu. Đảo ngược process ở inference. Ngày nay, mọi model hình ảnh, video, 3D và âm nhạc chính thống đều chạy trên vòng lặp này, có thể có các thủ thuật kết hợp luồng hoặc nhất quán ở trên cùng.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 3 · 02 (Backprop), Giai đoạn 8 · 02 (VAE)
**Thời lượng:** ~75 phút

## Vấn đề

Bạn muốn một bộ lấy mẫu cho `p_data(x)`. GAN chơi một trò chơi minimax thường phân kỳ. VAE tạo ra các mẫu mờ từ decoder Gaussian. Những gì bạn thực sự muốn là một mục tiêu training đó là (a) một loss ổn định duy nhất (không có điểm yên ngựa, không có minimax), (b) giới hạn dưới trên `log p(x)` (vì vậy bạn có likelihoods) và (c) các mẫu phù hợp với chất lượng SOTA.

Sohl-Dickstein et al. (2015) đã có một câu trả lời lý thuyết: xác định một `q(x_t | x_{t-1})` chuỗi Markov dần dần thêm nhiễu Gaussian và huấn luyện một `p_θ(x_{t-1} | x_t)` chuỗi ngược để khử nhiễu. Ho, Jain, Abbeel (2020) cho thấy loss có thể được đơn giản hóa thành một dòng - dự đoán nhiễu - và làm sạch phép toán. Vào năm 2020, đây là một sự tò mò. Vào năm 2021, nó đã sản xuất các mẫu hiện đại. Vào năm 2022, nó trở thành Khuếch tán ổn định. Vào năm 2026, nó là chất nền.

## Khái niệm

![DDPM: forward noise, reverse denoise](../assets/ddpm.svg)

**Chuyển tiếp process `q`.** Thêm nhiễu Gaussian trong `T` bước nhỏ. Hình thức đóng - lý do toán học có thể xử lý được - là bước tích lũy cũng là Gaussian:

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

nơi `α̅_t = ∏_{s=1..t} (1 - β_s)` cho một lịch trình `β_t`. Chọn `β_t` từ 1e-4 đến 0.02 tuyến tính trên T = 1000 bước và `x_T` là khoảng `N(0, I)`.

**Đảo ngược process `p_θ`.** Tìm hiểu một `ε_θ(x_t, t)` mạng nơ-ron dự đoán nhiễu đã được thêm vào. Cho `x_t`, khử nhiễu bằng cách:

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

trong đó `σ_t` là `sqrt(β_t)` hoặc là một variance có học thức. Biểu thức này xấu xí nhưng nó chỉ là đại số - giải cho `x_{t-1}` với posterior `q(x_{t-1} | x_t, x_0)` và thay thế `x_0` bằng ước tính dự đoán nhiễu của nó.

**Training loss.**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

Lấy mẫu `x_0` từ dữ liệu, chọn một `t` ngẫu nhiên, lấy mẫu `ε ~ N(0, I)`, tính toán `x_t` nhiễu trong một lần chụp thông qua dạng đóng và hồi quy về nhiễu. Một loss, không minimax, không KL, không có thủ thuật tham số lại.

**Sampling.** Bắt đầu `x_T ~ N(0, I)`. Lặp lại bước ngược lại từ `t = T` đến `1`. Xong.

## Tại sao nó hoạt động

Ba trực giác:

1. **Khử nhiễu rất dễ dàng; Tạo ra rất khó.** Ở `t=T`, dữ liệu là nhiễu thuần túy - mạng phải giải quyết một vấn đề nhỏ nhặt. Ở `t=0`, mạng chỉ phải dọn dẹp một vài pixel. Ở `t` trung cấp, vấn đề rất khó nhưng lưới có nhiều gradients chảy qua cùng một trọng lượng từ mọi mức độ nhiễu.

2. **Điểm số phù hợp trong ngụy trang.** Vincent (2011) đã chứng minh rằng dự đoán nhiễu tương đương với ước tính `∇_x log q(x_t | x_0)`, * điểm số *. SDE ngược lại sử dụng điểm số này để tăng mật độ gradient - một bước đi ngẫu nhiên có hướng dẫn đến các vùng có xác suất cao.

3. **ELBO giảm xuống MSE đơn giản.** Giới hạn dưới biến thiên đầy đủ có số hạng KL cho mỗi bước thời gian. Với tham số hóa DDPM, các thuật ngữ KL đó đơn giản hóa thành MSE về dự đoán nhiễu với các hệ số cụ thể; Ho đã bỏ các hệ số (gọi nó là loss "đơn giản") và chất lượng *cải thiện*.

```figure
diffusion-denoise
```

## Tự xây dựng

`code/main.py` triển khai DDPM 1-D. Dữ liệu là một hỗn hợp hai chế độ. "Net" là một MLP nhỏ lấy `(x_t, t)` và xuất ra nhiễu dự đoán. Training là loss một dòng. Sampling lặp lại chuỗi đảo ngược.

### Bước 1: lịch trình chuyển tiếp (mẫu đóng)

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### Bước 2: lấy mẫu `x_t` trong một lần chụp

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### Bước 3: một bước training

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### Bước 4: đảo ngược sampling

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

Đối với bài toán 1-D với 40 bước thời gian và MLP 24 đơn vị, điều này học hỗn hợp hai chế độ trong ~200 epochs.

## Điều kiện thời gian

Mạng cần biết nó đang khử nhiễu bước thời gian nào. Hai tùy chọn tiêu chuẩn:

- **embedding hình sin.** Giống như mã hóa vị trí Transformer. `embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`. Vượt qua MLP, phát vào lưới.
- **Điều hòa phim / định mức nhóm.** Dự án embedding scale/bias trên mỗi kênh (FiLM) tại mỗi khối.

Mã đồ chơi của chúng tôi sử dụng hình sin → concat. Production U-Nets sử dụng FiLM.

## Cạm bẫy

- **Lịch trình rất quan trọng.** `β` tuyến tính là mặc định của DDPM nhưng lịch trình cosin (Nichol & Dhariwal, 2021) cho FID tốt hơn cho cùng một tính toán. Chuyển đổi lịch trình nếu chất lượng ổn định.
- **embedding bước thời gian rất mong manh.** Truyền `t` thô như một phao hoạt động cho đồ chơi 1-D nhưng không thành công đối với hình ảnh; Luôn sử dụng embedding thích hợp.
- **Dự đoán V so với dự đoán ε.** Đối với các chế độ hẹp (t rất nhỏ hoặc rất lớn), `ε` có tín hiệu trên nhiễu kém. V-prediction (`v = α·ε - σ·x`) ổn định hơn; SDXL, SD3 và Flux sử dụng nó.
- **Classifier-free guidance.** Tại inference, tính cả `ε` có điều kiện và vô điều kiện, sau đó `ε_cfg = (1 + w) · ε_cond - w · ε_uncond` với `w ≈ 3-7`. Đề cập trong bài 08.
- **1000 bước là rất nhiều.** Production sử dụng DDIM (20-50 bước), DPM-Solver (10-20 bước) hoặc distillation (1-4 bước). Xem bài 12.

## Ứng dụng

| Vai trò | stack tiêu biểu năm 2026 |
|------|-----------------------|
| Khuếch tán không gian pixel hình ảnh (nhỏ, đồ chơi) | DDPM + U-Net |
| Hình ảnh khuếch tán tiềm ẩn | VAE encoder + U-Net hoặc DiT (Bài 07) |
| Khuếch tán tiềm ẩn video | DiT không gian thời gian (Sora, Veo, WAN) |
| Khuếch tán tiềm ẩn âm thanh | Bộ mã hóa + khuếch tán transformer |
| Khoa học (phân tử, protein, vật lý) | Khuếch tán tương đương (EDM, RFdiffusion, AlphaFold3) |

Khuếch tán là xương sống sinh sản phổ quát. Kết hợp dòng chảy (Bài 13) là đối thủ cạnh tranh 2024-2026 thường giành chiến thắng về tốc độ inference với cùng chất lượng.

## Sản phẩm bàn giao

Tiết kiệm `outputs/skill-diffusion-trainer.md`. Skill cần dataset + ngân sách và đầu ra tính toán: lịch trình (linear/cosine/sigmoid), mục tiêu dự đoán (ε/v/x), số bước, thang đo hướng dẫn, dòng bộ lấy mẫu và giao thức đánh giá.

## Bài tập

1. **Dễ dàng.** Thay đổi T từ 40 thành 10 trong `code/main.py`. Chất lượng mẫu (biểu đồ trực quan của đầu ra) giảm như thế nào? Cấu trúc hai chế độ sụp đổ ở T nào?
2. **Trung bình.** Chuyển từ dự đoán ε sang dự đoán v. Rút lại bước ngược lại. So sánh chất lượng mẫu cuối cùng.
3. **Khó.** Thêm classifier-free guidance. Điều kiện trên nhãn class `c ∈ {0, 1}`, thả nó 10% thời gian trong quá trình training và sampling lúc sử dụng `ε = (1+w)·ε_cond - w·ε_uncond`. Đo tỷ lệ trúng chế độ có điều kiện ở `w = 0, 1, 3, 7`.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Chuyển tiếp process | "Thêm nhiễu" | Chuỗi Markov cố định 'q (x_t \ | x_{t-1})' phá hủy dữ liệu. |
| Đảo ngược process | "Khử nhiễu" | Chuỗi học 'p_θ(x_{t-1} \ | x_t)' để xây dựng lại dữ liệu. |
| β lịch trình | "Thang nhiễu" | variance mỗi bước; tuyến tính, cosin hoặc sigmoid. |
| α̅ | "Thanh alpha" | `∏(1 - β)` sản phẩm tích lũy; cho `x_t` dạng đóng từ `x_0`. |
| loss đơn giản | "MSE về nhiễu" | `\ | \ | ε - ε_θ(x_t, t)\ | \ | ²`; tất cả các dẫn xuất biến thể đều sụp đổ thành điều này. |
| Dự đoán ε | "Dự đoán nhiễu" | Đầu ra là nhiễu được thêm vào; DDPM tiêu chuẩn. |
| Dự đoán chữ V | "Dự đoán vận tốc" | Đầu ra là `α·ε - σ·x`; điều hòa tốt hơn trên t. |
| DDPM | "Tờ giấy" | Ho và cộng sự. 2020; β tuyến tính, 1000 bước, U-Net. |
| DDIM | "Bộ lấy mẫu xác định" | Bộ lấy mẫu không phải Markov, 20-50 bước, cùng training mục tiêu. |
| Classifier-free guidance | "CFG" | Kết hợp các dự đoán nhiễu có điều kiện và vô điều kiện để khuếch đại điều hòa. |

## Lưu ý Production: inference khuếch tán là một vấn đề về số bước

Giấy DDPM chạy T = 1000 bước ngược lại. Không ai ships điều đó trong production. Mỗi inference stack thực sự chọn một trong ba chiến lược - và mỗi chiến lược ánh xạ rõ ràng production khung "độ trễ đến từ đâu":

1. **Bộ lấy mẫu nhanh hơn, cùng model.** DDIM (20-50 bước), DPM-Solver++ (10-20), UniPC (8-16). Thay thế vòng lặp ngược; trọng lượng `ε_θ` được huấn luyện không bị ảnh hưởng. Cắt giảm độ trễ 20-50×.
2. **Distillation.** Huấn luyện học sinh phù hợp với giáo viên trong ít bước hơn: Distillation lũy tiến (2 → 1), Models nhất quán (tùy ý → 1-4), LCM, SDXL-Turbo, SD3-Turbo. Giảm độ trễ thêm 5-10×, yêu cầu huấn luyện lại.
3. **Bộ nhớ đệm và biên dịch.** `torch.compile(unet, mode="reduce-overhead")`, phần phụ trợ khuếch tán của TensorRT-LLM, attention `xformers`/SDPA, trọng số bf16. Cắt giảm độ trễ trên mỗi bước ~2×. Stacks với (1) và (2).

Đối với server khuếch tán production, cuộc trò chuyện về ngân sách giống như production tài liệu mô tả cho LLMs: độ trễ là `num_steps × step_cost + VAE_decode`, thông lượng là `batch_size × (num_steps × step_cost)^-1`. TTFT nhỏ (một bước); Tương đương với TPOT là thời gian phản hồi đầy đủ vì việc tạo hình ảnh là "tất cả cùng một lúc" từ quan điểm của người dùng.

## Đọc thêm

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) - bài báo khuếch tán, đi trước thời đại.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM.
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) - DDIM, ít bước hơn.
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) - lịch trình cosine, đã học variance.
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — hướng dẫn về bộ phân loại.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) - CFG.
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) - ký hiệu thống nhất, công thức sạch nhất.
