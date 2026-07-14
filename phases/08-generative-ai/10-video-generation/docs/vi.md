# Tạo video

> Hình ảnh là tensor 2D. Video là video 3-D. Lý thuyết là như nhau; tính toán khó hơn 10-100 lần. Sora của OpenAI (tháng 2 năm 2024) đã chứng minh điều đó là có thể. Đến năm 2026, Veo 2, Kling 1.5, Runway Gen-3, Pika 2.0 và WAN 2.2 ship production video từ văn bản ở 1080p - và stack trọng lượng mở (CogVideoX, HunyuanVideo, Mochi-1, WAN 2.2) chậm hơn 12 tháng.

**Loại:** Xây dựng
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 8 · 07 (Khuếch tán tiềm ẩn), Giai đoạn 7 · 09 (ViT), Giai đoạn 8 · 06 (DDPM)
**Thời lượng:** ~45 phút

## Vấn đề

Video 1080p 10 giây ở tốc độ 24 khung hình / giây là 240 khung hình 1920×1080×3 pixel. Đó là ~1,5 GB dữ liệu thô cho mỗi clip. Khuếch tán không gian pixel là không khả thi. Bạn cần:

1. **Nén không gian.** VAE mã hóa video, không phải khung hình, thành một chuỗi các bản vá không gian-thời gian.
2. **Gắn kết thời gian.** Khung hình cần chia sẻ nội dung, ánh sáng và nhận dạng đối tượng trong vài giây. Mạng phải model chuyển động.
3. **Ngân sách tính toán.** Video training đắt hơn 10-100 lần so với hình ảnh có cùng kích thước model.
4. **Điều hòa.** Văn bản, hình ảnh (khung hình đầu tiên), âm thanh hoặc video khác. Hầu hết production models chấp nhận cả bốn.

Kiến trúc giải quyết vấn đề này là **Transformer khuếch tán (DiT)** được áp dụng cho các bản vá không gian, được huấn luyện trên datasets khổng lồ (prompt, chú thích, video). loss khuếch tán tương tự như Bài 06.

## Khái niệm

![Video diffusion: patchify, DiT, decode](../assets/video-generation.svg)

### Vá lỗi

Mã hóa video bằng VAE 3D (nén không gian đã học). Tiềm ẩn là hình dạng `[T_latent, H_latent, W_latent, C_latent]`. Chia thành các mảng có kích thước `[t_p, h_p, w_p]`. Đối với models kiểu Sora, `t_p = 1` (bản vá trên mỗi khung hình) hoặc `t_p = 2` (cứ sau hai khung hình). Video 1080p dài 10 giây nén thành ~20.000-100.000 bản vá.

### DiT không gian thời gian

A transformer processes chuỗi các bản vá phẳng. Mỗi bản vá có một embedding vị trí 3D (thời gian + y + x). Attention thường được phân tử:

- **attention không gian **trong các bản vá của mỗi khung hình.
- **attention thời gian **trên các khung hình tại cùng một vị trí không gian.
- **Full 3D attention **đắt hơn 16-100 lần; chỉ được sử dụng ở độ phân giải thấp hoặc trong nghiên cứu.

### Text conditioning

Cross-attention với một encoder văn bản lớn (T5-XXL cho Sora, CogVideoX-5B sử dụng T5-XXL). Vấn đề prompts lâu - bộ training của Sora có GPT-generated chú thích lại dày đặc trung bình 200 tokens mỗi clip.

### Training

loss khuếch tán chuẩn (dự đoán ε hoặc v) trên các tiềm ẩn không gian-thời gian. Dữ liệu: video web + ~100 triệu clip được tuyển chọn + chú thích văn bản tổng hợp. Điện toán: 10.000+ GPU giờ cho dù là một lần chạy nghiên cứu nhỏ; Thang Sora là 100.000+.

## Bối cảnh production 2026

| Model | Ngày | Thời lượng tối đa | Độ phân giải tối đa | Tạ mở? | Đáng chú ý |
|-------|------|--------------|---------|---------------|---------|
| Sora (OpenAI) | 2024-02 | Thập niên 60 | 1080 điểm | Không | model đầu tiên hiển thị các thuộc tính giả lập thế giới trên quy mô lớn |
| Sora Turbo | 2024-12 | Tuổi 20 | 1080 điểm | Không | Production Sora nhanh hơn 5 lần inference |
| Veo 2 (Google) | 2024-12 | 8 giây | 4 nghìn | Không | Chất lượng + vật lý cao nhất vào năm 2025 |
| Veo 3 | Quý 3 năm 2025 | 15 giây | 4 nghìn | Không | Âm thanh gốc và điều khiển máy ảnh mạnh mẽ hơn |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10 giây | 1080 điểm | Không | Chuyển động tốt nhất của con người vào năm 2025 Q1 |
| Đường băng Gen-3 Alpha | 2024-06 | 10 giây | 768 điểm | Không | Các công cụ video chuyên nghiệp trên đầu |
| Pika 2.0 | 2024-10 | 5 giây | 1080 điểm | Không | Tính nhất quán của nhân vật mạnh nhất |
| Bánh răng VideoX (THUDM) | 2024 | 10 giây | 720p | Có (2B, 5B) | Video tỷ lệ 5B mở đầu tiên |
| HunyuanVideo (Tencent) | 2024-12 | 5 giây | 720p | Có (13B) | Mở SOTA cuối năm 2024 |
| Mochi-1 (Genmo) | 2024-10 | 5,4 giây | 480p | Có (10B) | Được cấp phép cho phép nhất |
| WAN 2.2 (Alibaba) | 2025-07 | 5 giây | 720p | Có | Mở cửa mạnh nhất model giữa năm 2025 |

Trọng lượng mở đang thu hẹp khoảng cách nhanh hơn so với trong không gian hình ảnh: HunyuanVideo + WAN 2.2 LoRA đã cung cấp năng lượng cho hầu hết các quy trình làm việc mã nguồn mở vào giữa năm 2026.

## Tự xây dựng

`code/main.py` mô phỏng ý tưởng DiT không gian thời gian cốt lõi: vá một video tổng hợp nhỏ, thêm embedding vị trí mỗi bản vá và khử nhiễu toàn bộ chuỗi bằng attention kiểu transformer trên các bản vá. Không numpy; Python tinh khiết. Chúng ta chỉ ra rằng sự gắn kết thời gian xuất hiện ngay cả trong 1-D khi các bản vá khung liền kề chia sẻ bộ khử nhiễu và vị trí embeddings.

### Bước 1: vá "video" 1-D tổng hợp

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### Bước 2: embedding vị trí trên mỗi khung hình

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### Bước 3: khử nhiễu nhìn thấy toàn bộ trình tự

Thay vì khử nhiễu từng khung hình một cách độc lập, mạng nhỏ của chúng tôi nối tất cả các giá trị khung hình + vị trí của chúng embeddings và dự đoán nhiễu cho tất cả các khung hình cùng nhau.

### Bước 4: kiểm tra độ kết hợp thời gian

Sau khi training, hãy lấy mẫu video. Đo delta từ khung đến khung. Nếu model đã học được cấu trúc thời gian, các đồng bằng vẫn nhỏ hơn sampling mỗi khung một cách độc lập.

## Cạm bẫy

- **sampling trên mỗi khung hình độc lập = nhấp nháy.** Nếu bạn chạy khuếch tán hình ảnh trên từng khung hình riêng biệt, đầu ra sẽ nhấp nháy vì nhiễu của mỗi khung hình là độc lập. Khuếch tán video khắc phục điều này bằng cách ghép các khung hình thông qua nhiễu attention hoặc chia sẻ.
- **attention 3D ngây thơ = OOM.** attention 3D đầy đủ trên 10 giây 1080p tiềm ẩn là hàng trăm tỷ thao tác. Phân tích thành không gian + thời gian.
- **Phụ đề dữ liệu quan trọng hơn kích thước.** Nâng cấp chính của Sora trong prior tác phẩm là training phụ đề chi tiết hơn ~10 lần (GPT-4 clip được gắn nhãn lại). Báo cáo kỹ thuật của OpenAI rất rõ ràng về điều này.
- **Điều hòa khung hình đầu tiên.** Hầu hết các production models cũng chấp nhận hình ảnh làm khung hình đầu tiên. Đây là chế độ "hình ảnh thành video"; training bao gồm biến thể này.
- **Trôi vật lý.** Các clip dài (>10 giây) tích lũy những mâu thuẫn tinh tế. Tạo cửa sổ trượt + neo khung hình chính giúp ích.

## Ứng dụng

| Trường hợp sử dụng | Lựa chọn năm 2026 |
|----------|-----------|
| Chuyển văn bản thành video chất lượng cao nhất, được lưu trữ | Veo 3 hoặc Sora |
| Điện ảnh điều khiển bằng máy ảnh | Runway Gen-3 với bàn chải chuyển động |
| Tính nhất quán của nhân vật trên các clip | Pika 2.0 hoặc Kling 2.1 |
| Tạ mở, fine-tune nhanh | WAN 2.2 + LoRA |
| Chuyển hình ảnh thành video | WAN 2.2-I2V, Kling 2.1 I2V hoặc Đường băng |
| Đồng bộ hóa môi âm thanh sang video | Veo 3 (âm thanh gốc) hoặc model hát nhép chuyên dụng |
| Chỉnh sửa video | Runway Act-Two, Kling Motion Brush, Flux-Kontext (khung hình tĩnh) |

Chi phí mỗi giây của video ở mức chất lượng ngang bằng đã giảm 20 lần từ năm 2024 đến năm 2026.

## Sản phẩm bàn giao

Tiết kiệm `outputs/skill-video-brief.md`. Skill tóm tắt video (thời lượng, tỷ lệ khung hình, kiểu dáng, sơ đồ máy ảnh, tính nhất quán của đối tượng, âm thanh) và đầu ra: model + lưu trữ, giàn giáo prompt (ngôn ngữ máy ảnh, mô tả chủ đề, mô tả chuyển động), giao thức hạt giống + khả năng tái tạo và danh sách kiểm tra QA cấp khung hình.

## Bài tập

1. **Dễ dàng.** Trong `code/main.py`, hãy so sánh delta giữa các khung hình cho (a) sampling trên mỗi khung hình độc lập, (b) sampling trình tự chung. Báo cáo trung bình và variance của đồng bằng.
2. **Trung bình.** Thêm điều kiện khung hình đầu tiên: ghim khung hình 0 vào một giá trị nhất định và lấy mẫu rest. Đo lường cách giá trị được ghim lan truyền.
3. **Khó.** Sử dụng bộ khuếch tán HuggingFace để chạy CogVideoX-2B trên GPU cục bộ. Thời gian 20 inference bước ở 720p cho clip dài 6 giây. Lập hồ sơ attention không gian thời gian để xác định nút thắt cổ chai.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| Video VAE | "VAE 3-D" | Encoder nén `(T, H, W, C)` → tiềm ẩn không gian thời gian. |
| Bản vá lỗi | "Người tokens" | Các khối 3-D có kích thước cố định của tiềm ẩn; đầu vào DiT. |
| attention phân tích | "Không gian + thời gian" | Chạy attention trên không gian, sau đó theo thời gian; bỏ qua toàn bộ attention 3D. |
| Chuyển hình ảnh thành video (I2V) | "Tạo hiệu ứng cho ảnh này" | Model lấy hình ảnh + văn bản, xuất video bắt đầu từ nó. |
| Điều hòa khung hình chính | "Khung neo" | Ghim các khung hình cụ thể để kiểm soát vòng cung của video. |
| Bàn chải chuyển động | "Gợi ý định hướng" | Đầu vào giao diện người dùng nơi người dùng vẽ vectors chuyển động lên hình ảnh. |
| Chú thích lại | "Chú thích dày đặc" | Sử dụng LLM để gắn nhãn lại các clip training với prompts chi tiết. |
| Nhấp nháy | "artifact thời gian" | Sự không nhất quán giữa khung hình; cố định với khử nhiễu kết hợp. |

## Lưu ý Production: Tiềm ẩn video là một vấn đề về băng thông bộ nhớ

Clip 1080p dài 10 giây ở tốc độ 24 khung hình / giây là 240 khung hình × 1920 × 1080 × 3 ≈ 1.5 GB pixel thô. Sau khi nén VAE video 4× (`2 × spatial × 2 × temporal`), tiềm ẩn là ~100 MB cho mỗi yêu cầu. Chạy điều này qua một DiT không gian trong 30 bước ở batch 1 và bạn đang di chuyển ~3 GB/step qua HBM - băng thông bộ nhớ, không phải FLOPs, là nút thắt cổ chai.

Ba nút production, tất cả đều trực tiếp từ văn học production inference inference chương:

- **TP trên các models Chuyển văn bản thành video của DiT.** thường được ≥10 tỷ tham số. TP = 4 trên 4 H100 là tiêu chuẩn; PP = 2 × TP = 2 cho 405B-class models. Độ trễ trên mỗi bước giảm gần như tuyến tính với TP lên đến bức tường giảm hoàn toàn.
- **Frame batching = liên tục batching.** Tại thời điểm tạo, video về mặt khái niệm là một batch các khung hình được liên kết bởi attention. Áp dụng tính năng phân lô liên tục (lập lịch trong chuyến bay): bắt đầu hiển thị `t+1` khung hình trong khi `t-1` khung đang được trả về, nếu kiến trúc model cho phép tạo cửa sổ trượt.
- **Bộ nhớ đệm điền trước ở cấp độ clip.** Đối với hình ảnh thành video, điều hòa khung hình đầu tiên tương tự như tính năng điền trước prompt của LLM: tính toán một lần, sử dụng lại qua các decoder thời gian. Đây thực sự là một bộ nhớ đệm KV cho video.

## Đọc thêm

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Báo cáo kỹ thuật của Sora.
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) - CogVideoX.
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo.
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1.
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) - mở cửa SOTA giữa năm 2025.
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) - bài báo phổ biến video quan trọng.
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — Tổ tiên của Stable Video Diffusion.
