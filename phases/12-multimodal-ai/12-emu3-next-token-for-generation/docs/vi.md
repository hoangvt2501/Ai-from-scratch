# Emu3: Dự đoán Token tiếp theo cho việc tạo hình ảnh và video

> Emu3 của BAAI (Wang và cộng sự, tháng 9 năm 2024) là kết quả năm 2024 lẽ ra phải kết thúc cuộc tranh luận khuếch tán so với tự hồi quy. Một transformer chỉ decoder kiểu Llama duy nhất, chỉ được huấn luyện trên mục tiêu dự đoán token tiếp theo, trên một từ vựng thống nhất gồm văn bản + tokens hình ảnh VQ + tokens video VQ 3D, đánh bại SDXL về tạo hình ảnh và LLaVA-1.6 về nhận thức. Không có loss CLIP. Không có lịch khuếch tán. Classifier-free guidance được sử dụng ở mức inference cho chất lượng, nhưng mục tiêu cốt lõi của training là dự đoán token tiếp theo với sự ép buộc của giáo viên. Được xuất bản trên Nature. Bài học này đọc luận điểm Emu3 - tại sao tỷ lệ tokenizer cộng tốt hơn là tất cả những gì bạn cần - và tương phản với các phương pháp khuếch tán.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, 3D video tokenizer math + autoregressive sampler skeleton)
**Kiến thức tiên quyết:** Giai đoạn 12 · 11 (Tắc kè hoa)
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Giải thích lý do tại sao mục tiêu một loss token tiếp theo của Emu3 hoạt động bất chấp giả định từ lâu rằng cần phải khuếch tán để đảm bảo chất lượng hình ảnh.
- Mô tả tokenizer video 3D: sách mã VQ không gian thời gian trông như thế nào, tại sao lại có các bản vá span thời gian.
- So sánh Emu3 và Stable Diffusion XL trên (training tính toán, chi phí inference, trần chất lượng).
- Kể tên ba vai trò mà cùng một Emu3 model đóng: Emu3-Gen (tạo hình ảnh), Emu3-Chat (nhận thức), Emu3-Stage2 (tạo video).

## Vấn đề

Sự khôn ngoan thông thường đến năm 2024: tạo hình ảnh cần được phổ biến. Lập luận: hình ảnh rời rạc tokens mất quá nhiều thông tin để xây dựng lại chi tiết và sampling tự hồi quy tích lũy lỗi trên hàng nghìn tokens. Stable Diffusion, DALL-E 3, Imagen, Midjourney đều sử dụng một số hình thức khuếch tán. Chameleon (Bài 12.11) đã bác bỏ một phần điều này ở quy mô nhỏ nhưng không phù hợp với SDXL về chất lượng.

Emu3 tấn công trực diện vào cuộc tranh luận. Tuyên bố: tokenizer hình ảnh tốt hơn + đủ tỷ lệ + token loss tiếp theo = tạo hình ảnh đánh bại khuếch tán trong cùng một model cũng thực hiện nhận thức.

Vụ đặt cược đã gây tranh cãi khi nó được công bố. Hai năm sau, họ thế hệ thống nhất mã nguồn mở (Emu3, Show-o, Janus-Pro, Transfusion) là con đường mặc định cho nghiên cứu; production models biên giới dường như sử dụng một số biến thể.

## Khái niệm

### The Emu3 tokenizer

Thành phần quan trọng là tokenizer hình ảnh. Emu3 huấn luyện IBQ-class tokenizer tùy chỉnh (Inverse Bottleneck Quantizer, dòng SBER-MoVQGAN) ở mức giảm độ phân giải 8x8 mỗi token. Hình ảnh 512x512 trở thành 64x64 = 4096 tokens ở kích thước sách mã 32768.

Con số này lớn hơn 1024 tokens trên 512x512 của Chameleon ở K = 8192 nhưng rẻ hơn mỗi token (tra cứu sách mã nhỏ hơn, codec đơn giản hơn). Chỉ số chính: tái tạo PSNR ở mức 30,5 dB, cạnh tranh với không gian tiềm ẩn liên tục của Stable Diffusion ở mức 32 dB.

Đối với video: tokenizer VQ 3D mã hóa một bản vá không gian thời gian (4x4x4 pixel) thành một số nguyên. Clip 4 giây ở tốc độ 8 FPS có 32 khung hình; ở 256x256 với 4x không gian và 4x giảm thời gian, số lượng token là (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 tokens.

Chất lượng Tokenizer là trần nhà. Đóng góp của Emu3 một phần là "chúng tôi đã huấn luyện một tokenizer rất tốt".

### Một loss training

Emu3 sử dụng một mục tiêu: dự đoán token tiếp theo về từ vựng được chia sẻ trên tokens văn bản, tokens hình ảnh 2D và tokens video 3D. Trọng số được nhân với các yếu tố cụ thể về phương thức trong quá trình training để cân bằng đóng góp, nhưng hàm loss giống hệt nhau.

Huấn luyện kết hợp của:
- Thế hệ hình ảnh: `<text caption> <image> image_tokens </image>`
- Cảm nhận hình ảnh: `<image> image_tokens </image> <question> text_tokens`
- Thế hệ video: `<text caption> <video> video_tokens </video>`
- Cảm nhận video: tương tự.
- Chỉ văn bản: NTP tiêu chuẩn.

model tìm hiểu khi nào nên phát tokens hình ảnh so với tokens văn bản từ phân phối dữ liệu. Thế hệ xuất hiện từ model dự đoán hình ảnh tokens sau thẻ `<image>`.

### Classifier-free guidance và temperature

Tạo hình ảnh tự hồi quy trở nên tốt hơn nhiều với classifier-free guidance (CFG) ở inference. Emu3 sử dụng nó: tạo hai lần, một lần với chú thích đầy đủ, một lần với chú thích trống, trộn logits với trọng lượng hướng dẫn (điển hình 3.0-7.0). Đây là cùng một cách sử dụng khuếch tán thủ thuật CFG, vay mượn cho cài đặt tự hồi quy.

Temperature vấn đề: quá cao, artifacts; quá thấp, chế độ thu gọn. temperature đề xuất của Emu3 là 1.0 cho nhận thức, 0.8 cho tạo hình ảnh.

### Ba vai trò, một model

Emu3 ships ba APIs khác biệt về chức năng nhưng là một tập hợp trọng lượng cơ bản:

- Emu3-Gen. Tạo hình ảnh. Văn bản đầu vào, hình ảnh đầu ra tokens.
- Emu3-Trò chuyện. VQA và chú thích. Hình ảnh đầu vào (tokens), văn bản đầu ra.
- Emu3-Giai đoạn2. Tạo video và VQA video. Nhập văn bản hoặc video, xuất văn bản hoặc video.

Không có người đứng đầu nhiệm vụ cụ thể. Chỉ là các mẫu prompt khác nhau. Tương tự checkpoint.

### Benchmarks

Từ bài báo Emu3 (Tháng 9 năm 2024):

- Tạo hình ảnh: đánh bại SDXL trên MJHQ-30K FID (5,4 so với 5,6), GenEval tổng thể (0,54 so với 0,55 - hòa thống kê) và tổng hợp của Deep-Eval.
- Nhận thức hình ảnh: đánh bại LLaVA-1.6 trên VQAv2 (75.1 so với 72.4) và gần như khớp với MMMU.
- Tạo video: Chất lượng clip 4 giây tại FVD cạnh tranh với models chuẩn công khai thời Sora.

Các con số không phải lúc nào cũng chiến thắng - Emu3 đánh đổi một điểm ở đây để lấy một điểm ở đó - nhưng tuyên bố "dự đoán token tiếp theo là tất cả những gì bạn cần" có thể được bảo vệ trên các phương thức.

### Chi phí điện toán

Emu3 đã được huấn luyện trên ~300 tỷ tokens đa phương thức với 7B-parameter model. GPU giờ gần tương đương với pretraining Llama-2-7B (2k-4k GPU năm trên silicon A100-class). Các models khuếch tán như Stable Diffusion 3 huấn luyện với ngân sách tương tự nhưng cần encoders văn bản riêng biệt và pipelines phức tạp hơn.

Ở mức inference, Emu3 chậm hơn SDXL trên mỗi hình ảnh: 4096 hình ảnh tokens ở 30 tok/s là ~2 phút trên mỗi hình ảnh 512x512, so với 2-5 giây đối với SDXL. Giải mã suy đoán và tối ưu hóa bộ nhớ đệm KV thu hẹp khoảng cách nhưng không thu hẹp khoảng cách. Tạo hình ảnh tự hồi quy nặng về tính toán; Đây là sự đánh đổi thường trực.

### Tại sao điều này lại quan trọng

Đóng góp sâu sắc của Emu3 là khái niệm. Nếu dự đoán token tiếp theo thay đổi tỷ lệ phù hợp với sự khuếch tán khi tạo hình ảnh, thì đường dẫn model thống nhất (một loss, một đường trục, bất kỳ phương thức nào) là khả thi. Các models trong tương lai không cần encoders văn bản riêng biệt, bộ lập lịch khuếch tán riêng biệt, VAE riêng biệt. Một transformer, một tokenizer cho mỗi phương thức, quy mô.

Show-o, Janus-Pro và InternVL-tất cả các bạn đều xây dựng hoặc thách thức luận án này. Các phòng thí nghiệm của Trung Quốc (BAAI, DeepSeek) công bố theo hướng này tích cực hơn so với các phòng thí nghiệm của Mỹ cho đến năm 2025.

## Ứng dụng

`code/main.py` xây dựng hai món đồ chơi:

- Máy tính số lượng tokenizer VQ 2D so với 3D: được đưa ra (độ phân giải, bản vá, clip_length, FPS), tính toán token đếm cho hình ảnh và video.
- Một bộ lấy mẫu token hình ảnh tự hồi quy với classifier-free guidance ở temperature.

Việc triển khai CFG phù hợp với công thức của Emu3 - kết hợp logits có điều kiện và vô điều kiện với trọng số hướng dẫn.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-token-gen-cost-analyzer.md`. Với thông số kỹ thuật sản phẩm thế hệ (hình ảnh hoặc video, độ phân giải mục tiêu, bậc chất lượng, ngân sách độ trễ), nó tính toán số lượng token, chi phí inference và chọn dòng Emu3 so với khuếch tán.

## Bài tập

1. Emu3 tạo ra 4096 tokens trên mỗi hình ảnh 512x512 ở mức giảm 8x8. Tính toán giá trị tương đương cho 1024x1024 và 2048x2048. Điều gì xảy ra với độ trễ inference?

2. Đọc Emu3 Phần 3.3 trên video tokenizer. Mô tả hình dạng bản vá VQ 3D và lý do tại sao nó là 4x4x4 không phải 8x8x1.

3. Trọng lượng Classifier-free guidance 5.0 so với 3.0: Hiệu ứng hình ảnh gì? Trace toán học trong `code/main.py`.

4. Tính toán training FLOPs cho Emu3-7B ở 300B tokens và so sánh với Khuếch tán ổn định 3. Cái nào đắt hơn để huấn luyện?

5. Emu3 đánh bại SDXL trên FID nhưng không phải trên VQAv2 so với VLMs chuyên dụng. Giải thích lý do tại sao cách tiếp cận loss thống nhất cho thấy những điểm mạnh khác nhau so với các chuyên gia về các benchmarks khác nhau.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Dự đoán token tiếp theo | "NTP" | loss tự hồi quy tiêu chuẩn: dự đoán token[i+1] cho token[0..i]; Hoạt động cho mọi phương thức khi được mã hóa |
| IBQ tokenizer | "Bộ định lượng cổ chai nghịch đảo" | Một class của VQ-VAE với sách mã lớn hơn (32768+) và tái tạo tốt hơn của Chameleon |
| VQ 3D | "Bộ định lượng không gian" | Sổ mã được lập chỉ mục theo (thời gian, hàng, col); Một token bao phủ một khối lập phương 4x4x4 pixel |
| Classifier-free guidance | "CFG" | Trộn logits có điều kiện và vô điều kiện với gamma trọng lượng; Tăng chất lượng hình ảnh ở mức inference |
| Từ vựng thống nhất | "Chia sẻ tokens" | Văn bản + hình ảnh + video đều được vẽ từ cùng một không gian số nguyên; model dự đoán phương thức nào sẽ đến tiếp theo |
| MJHQ-30K | "Thế hệ hình ảnh benchmark" | benchmark chất lượng giữa hành trình với 30k prompts; Emu3 báo cáo FID tại đây |

## Đọc thêm

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
