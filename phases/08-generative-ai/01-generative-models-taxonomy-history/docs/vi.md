# Generative Models - Phân loại và Lịch sử

> Mọi model hình ảnh, model văn bản, model video và model 3D đều nằm gọn trong một trong năm nhóm. Chọn sai xô và bạn sẽ chiến đấu với phép toán trong nhiều tuần. Chọn đúng và mười hai năm tiến bộ cuối cùng của lĩnh vực này stacks rõ ràng trong đầu bạn.

**Loại:** Học
**Ngôn ngữ:** Python
**Kiến thức tiên quyết:** Giai đoạn 2 (ML Nguyên tắc cơ bản), Giai đoạn 3 (Deep Learning Core), Giai đoạn 7 · 14 (Transformers)
**Thời lượng:** ~45 phút

## Vấn đề

Một model tổng quát thực hiện một công việc: cho training mẫu được rút ra từ một số `p_data(x)` phân phối không xác định, xuất ra các mẫu mới trông giống như chúng đến từ cùng một phân phối. Khuôn mặt, câu, tệp MIDI, cấu trúc protein - tất cả đều giống nhau nếu bạn nheo mắt.

Vấn đề là `p_data` sống trong một không gian có hàng triệu chiều (hình ảnh RGB 512x512 là ~786k kích thước), các mẫu nằm trên một ống góp mỏng bên trong không gian đó và bạn chỉ có thể có 10 triệu ví dụ. Cưỡng bức mật độ là vô vọng. Mỗi model tổng quát là một sự thỏa hiệp đánh đổi một vấn đề khó cho một vấn đề ít khó hơn một chút.

Năm gia đình đã sống sót qua mười hai năm qua. Biết được sự thỏa hiệp nào mà mỗi gia đình thực hiện cho bạn biết tại sao nó thắng trong một số nhiệm vụ và sụp đổ trong những nhiệm vụ khác.

## Khái niệm

![Five families of generative models — taxonomy by what they model](../assets/taxonomy.svg)

**1. Mật độ rõ ràng, dễ xử lý.** Viết `log p(x)` như một tổng mà bạn thực sự có thể đánh giá. models tự hồi quy (PixelCNN, WaveNet, GPT) phân tích `p(x) = ∏ p(x_i | x_<i)`. Chuẩn hóa luồng (RealNVP, Glow) xây dựng `p(x)` như một phép biến đổi đảo ngược của một cơ sở đơn giản. Ưu điểm: likelihood chính xác, training loss sạch. Nhược điểm: inference tự hồi quy là tuần tự (chậm đối với các chuỗi dài), dòng chảy cần kiến trúc đảo ngược (hạn chế về mặt kiến trúc).

**2. Mật độ rõ ràng, gần đúng.** Liên kết `log p(x)` từ bên dưới (ELBO) và tối ưu hóa giới hạn. VAE (Kingma 2013) sử dụng encoder-decoder với posterior biến thể. models khuếch tán (DDPM, Ho 2020) huấn luyện một bộ khử nhiễu ngầm tối ưu hóa ELBO có trọng số. Khuếch tán là xương sống hình ảnh, video và 3D thống trị vào năm 2026.

**3. Mật độ ngầm.** Bỏ qua mật độ hoàn toàn; Tìm hiểu một `G(z)` máy phát điện tạo ra các mẫu và một `D(x)` phân biệt phân biệt giữa thật và giả. GAN (Goodfellow 2014). Nhịn ăn ở inference (một forward pass) nhưng nổi tiếng là không ổn định trong training. StyleGAN 1/2/3 vẫn là hiện đại cho chủ nghĩa ảnh chân thực miền cố định (khuôn mặt, phòng ngủ) ngay cả vào năm 2026.

**4. Dựa trên điểm số / thời gian liên tục.** Tìm hiểu trực tiếp gradient của `∇_x log p(x)` mật độ nhật ký (điểm số). Song & Ermon (2019) cho thấy sự phù hợp với điểm số khái quát hóa sự khuếch tán thành SDE. Kết hợp dòng chảy (Lipman 2023) là độ nóng của giai đoạn 2024-2026: training không mô phỏng, đường dẫn thẳng hơn, nhanh hơn 4-10 lần sampling so với DDPM. Stable Diffusion 3, Flux, AudioCraft 2 đều sử dụng kết hợp luồng.

**5. Tự hồi quy dựa trên Token trên các mã rời rạc.** Nén dữ liệu có độ mờ cao bằng VQ-VAE hoặc bộ lượng tử dư thành một chuỗi tokens rời rạc ngắn, sau đó sử dụng Transformer để model trình tự token. Parti, MuseNet, AudioLM, VALL-E, bản vá của Sora đều tokenizer sử dụng điều này. Đây là nhóm 1 cộng với một tokenizer đã học.

## Sơ lược về lịch sử

| Năm | Model | Tại sao điều này lại quan trọng |
|------|-------|-----------------|
| 2013 | VAE (Kingma) | Đầu tiên là model tổng quát sâu với một training loss có thể sử dụng được. |
| 2014 | GAN (Người bạn tốt) | Mật độ ngầm, không likelihood - các mẫu sắc nét đáng kinh ngạc. |
| 2015 | HÒA, PixelCNN | Tạo hình ảnh tuần tự. |
| 2017 | Phát sáng, RealNVP | Dòng chảy đảo ngược; likelihood chính xác với chiều sâu. |
| 2017 | GAN lũy tiến | Mặt megapixel đầu tiên. |
| 2019 | Phong cáchGAN / Phong cách GAN2 | Những khuôn mặt chân thực vẫn khó bị đánh bại đối với một lĩnh vực đó. |
| 2020 | DDPM (Hồ) | Sự khuếch tán trở nên thực tế. |
| 2021 | KẸP, DALL-E 1, VQGAN | Chuyển văn bản thành hình ảnh trở thành xu hướng chủ đạo. |
| 2022 | Hình ảnh, khuếch tán ổn định 1, DALL-E 2 | Khuếch tán tiềm ẩn + text conditioning = hàng hóa. |
| 2022 | ControlNet, LoRA | Kiểm soát tốt sự khuếch tán pretrained. |
| 2023 | SDXL, Midjourney v5, Kết hợp luồng | Quy mô + động lực training tốt hơn. |
| 2024 | Sora, Khuếch tán ổn định 3, Flux.1 | Khuếch tán video; Flow matching thắng. |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | Video cấp Production. |
| 2026 | Tính nhất quán + Dòng chảy chỉnh lưu | sampling một bước từ xương sống khuếch tán. |

## Phân loại năm câu hỏi

Khi một bài báo model tổng quát mới được tung ra, hãy trả lời năm câu hỏi này trước khi đọc phần phương pháp.

1. **Những gì đang được mô hình hóa?** Pixel, tiềm ẩn, tokens rời rạc, Gaussian 3D, lưới, dạng sóng?
2. **Mật độ là rõ ràng hay ngầm?** Họ có viết ra `log p(x)`?
3. **Sampling: one-shot hay lặp lại?** Lặp lại có nghĩa là inference chậm hơn; one-shot thường có nghĩa là đối nghịch hoặc chưng cất.
4. **Điều kiện: vô điều kiện, class, văn bản, hình ảnh, tư thế?** Điều này xác định loss và giàn giáo kiến trúc.
5. **Đánh giá: FID, điểm CLIP, IS, sở thích của con người, nhiệm vụ accuracy?** Mỗi chế độ đều có chế độ lỗi đã biết (xem Bài 14).

Bạn sẽ trả lời lại năm câu hỏi này cho mỗi bài học trong giai đoạn này. Cuối cùng, họ sẽ phản xạ.

## Tự xây dựng

Mã cho bài học này là một hình ảnh trực quan nhẹ: phù hợp với hỗn hợp 1-D của Gaussian từ các mẫu bằng cách sử dụng ba cách tiếp cận đồ chơi (mật độ hạt nhân, biểu đồ rời rạc và trình tạo "GAN-ish" mẫu gần nhất) để bạn có thể thấy sự khác biệt giữa mật độ rõ ràng và mật độ ngầm trên một vấn đề mà bạn có thể in trên một màn hình.

Chạy `code/main.py`. Nó lấy 2000 mẫu từ hỗn hợp Gaussian hai chế độ, sau đó in:

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

Lưu ý: hai câu đầu tiên cho phép bạn hỏi "khả năng điểm này là bao nhiêu?" Người thứ ba không thể. Đây là sự khác biệt * rõ ràng và ngầm * sẽ quan trọng đối với mọi bài học trong tương lai.

## Ứng dụng

Gia đình nào, cho nhiệm vụ nào, vào năm 2026?

| Nhiệm vụ | Gia đình tốt nhất | Tại sao |
|------|-------------|-----|
| Khuôn mặt chân thực, miền hẹp | StyleGAN 2/3 | Vẫn sắc nét nhất, nhanh nhất inference. |
| Chuyển văn bản thành hình ảnh chung | Khuếch tán tiềm ẩn + khớp dòng chảy | SD3, Thông lượng.1, DALL-E 3. |
| Chuyển văn bản thành hình ảnh nhanh chóng | Lưu lượng chỉnh lưu + distillation | SDXL-Turbo, SD3-Turbo, LCM. |
| Chuyển văn bản thành video | Khuếch tán Transformer + kết hợp dòng chảy | Sora, Veo 2, Kling. |
| Bài phát biểu + âm nhạc | AR dựa trên Token (AudioLM, VALL-E, MusicGen) hoặc kết hợp luồng (AudioCraft 2) | Quy mô tokens rời rạc với giá rẻ. |
| Cảnh 3D | Gaussian Splatting vừa vặn, khuếch tán prior | 3D-GS để tái tạo, khuếch tán cho góc nhìn mới lạ. |
| Ước tính mật độ (không sampling) | Dòng chảy | Chỉ có gia đình có `log p(x)` chính xác. |
| Mô phỏng / vật lý | Đối sánh luồng, điểm SDE | Đường thẳng, trường vector mịn. |

## Sản phẩm bàn giao

Lưu dưới dạng `outputs/skill-model-chooser.md`.

skill mô tả tác vụ và kết quả: (1) sử dụng dòng nào, (2) danh sách xếp hạng gồm ba tùy chọn mở và ba tùy chọn được lưu trữ, (3) chế độ lỗi có thể xảy ra mà bạn nên theo dõi và (4) ngân sách compute/time.

## Bài tập

1. **Dễ dàng.** Đối với mỗi sản phẩm trong số năm sản phẩm này, hãy xác định họ và xương sống: hình ảnh ChatGPT, Midjourney v7, Sora, Runway Gen-3, ElevenLabs. Bằng chứng phải là từ các báo cáo kỹ thuật công khai.
2. **Trung bình.** Bài báo bạn sắp đọc vào ngày mai tuyên bố sampling nhanh hơn 100 lần so với khuếch tán. Viết ra ba câu hỏi để kiểm tra xem tăng tốc có tồn tại sau điều kiện và độ phân giải cao hay không.
3. **Khó.** Lấy một lĩnh vực bạn quan tâm (ví dụ: cấu trúc protein, CAD, phân tử, quỹ đạo). Trả lời phân loại năm câu hỏi cho model SOTA hiện tại trong lĩnh vực đó và phác thảo những gì tốt hơn model sẽ thay đổi.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|-----------------------|
| model tổng quát | "Nó tạo ra những thứ mới" | Học một bộ lấy mẫu cho `p_data(x)`, tùy chọn hiển thị `log p(x)`. |
| Mật độ rõ ràng | "Bạn có thể đánh giá nó" | Model cung cấp một `log p(x)` dạng khép kín hoặc có thể điều khiển được. |
| Mật độ ngầm | "Phong cách GAN" | Chỉ có một bộ lấy mẫu - không có cách nào để đánh giá `p(x)` của một điểm nhất định. |
| ELBO | "Bằng chứng giới hạn dưới" | Một giới hạn dưới có thể điều khiển được trên `log p(x)`; VAE và khuếch tán tối ưu hóa nó. |
| Điểm số | "Gradient mật độ log" | `∇_x log p(x)`; khuếch tán và SDE models học lĩnh vực này. |
| Giả thuyết đa tạp | "Dữ liệu tồn tại trên một bề mặt" | Dữ liệu độ mờ cao tập trung vào một đa tạp độ mờ thấp; Tại sao giảm chiều hoạt động. |
| Tự hồi quy | "Dự đoán quân cờ tiếp theo" | Thừa số khớp như tích của các điều kiện. |
| Tiềm ẩn | "Mã nén" | Biểu diễn độ mờ thấp mà từ đó decoder có thể tái tạo đầu vào. |

## Production lưu ý: năm gia đình, năm hình dạng inference

Mỗi gia đình ánh xạ đến một đường cong chi phí inference-server khác nhau. Khung tài liệu production inference LLM inference dưới dạng điền trước + giải mã; Sự phân hủy tương tự cũng áp dụng ở đây:

- **Tự hồi quy (nhóm 1 và 5).** Giải mã tuần tự chiếm ưu thế về độ trễ; KV-cache, hàng loạt liên tục và giải mã đầu cơ đều áp dụng trực tiếp.
- **VAE / khuếch tán / khớp luồng (nhóm 2 và 4).** Không có giải mã theo nghĩa LLM. Chi phí = `num_steps × step_cost` và `step_cost` là chuyển tiếp transformer hoặc U-Net ở độ phân giải tiềm ẩn đầy đủ. Các nút production là đếm bước (DDIM / DPM-Solver / distillation), kích thước batch và precision (bf16 / fp8 / int4).
- **GAN (xô 3).** Một forward pass. Không có lịch trình, không có bộ nhớ đệm KV. TTFT ≈ tổng độ trễ. Đây là lý do tại sao StyleGAN vẫn giành chiến thắng trên UX miền hẹp.

Khi bạn thấy "nhanh hơn khuếch tán" trong bản tóm tắt bài báo, hãy dịch nó thành "ít bước hơn × chi phí cùng bước" hoặc "cùng bước × chi phí bước rẻ hơn". Mọi thứ khác đều là tiếp thị.

## Đọc thêm

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) - bài báo GAN.
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - bài báo VAE.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - bài báo DDPM.
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — khuếch tán dưới dạng SDE.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — giấy khớp dòng chảy.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) - Khuếch tán ổn định 3.
