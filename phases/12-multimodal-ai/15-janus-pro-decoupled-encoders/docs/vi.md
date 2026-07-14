# Janus-Pro: Encoders tách rời cho Models đa phương thức hợp nhất

> Các models đa phương thức thống nhất có một sự căng thẳng không thể tránh khỏi. Hiểu muốn features ngữ nghĩa - đầu ra SigLIP hoặc DINOv2 vectors phong phú với thông tin cấp khái niệm. Thế hệ muốn các mã thân thiện với tái tạo - VQ tokens kết hợp trở lại thành các pixel sắc nét. Hai mục tiêu không tương thích trong một encoder. Janus (DeepSeek, tháng 10 năm 2024) và Janus-Pro (DeepSeek, tháng 1 năm 2025) lập luận rằng cách khắc phục là ngừng cố gắng: tách hai encoders. Chia sẻ cơ thể transformer giữa các nhiệm vụ, nhưng định tuyến sự hiểu biết thông qua SigLIP và tạo thông qua tokenizer VQ. Ở 7B, Janus-Pro đánh bại DALL-E 3 trên GenEval trong khi phù hợp với LLaVA trên MMMU. Bài học này đọc lý do tại sao hai encoders hoạt động khi một người thất bại.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, dual-encoder routing + shared-body signal)
**Kiến thức tiên quyết:** Giai đoạn 12 · 13 (Truyền máu), Giai đoạn 12 · 14 (Hiển thị)
**Thời lượng:** ~120 phút

## Mục tiêu học tập

- Giải thích lý do tại sao một encoder được chia sẻ duy nhất ảnh hưởng đến sự hiểu biết hoặc chất lượng thế hệ.
- Mô tả định tuyến của Janus-Pro: SigLIP features ở phía đầu vào để hiểu, VQ tokens trên cả đầu vào và đầu ra để phát điện.
- Trace việc mở rộng quy mô hỗn hợp dữ liệu giúp Janus-Pro thành công trong khi Janus không làm được.
- So sánh các kiến trúc tách rời (Janus-Pro), liên tục ghép nối (Truyền máu) và rời rạc ghép nối (Show-o).

## Vấn đề

models thống nhất chia sẻ một cơ thể transformer qua sự hiểu biết và thế hệ. Các lần thử trước đó (Tắc kè hoa, Show-o, Truyền máu) đều sử dụng một tokenizer hình ảnh cho cả hai hướng. tokenizer là một sự thỏa hiệp:

- Tối ưu hóa để tái tạo (tạo): VQ-VAE ghi lại chi tiết điểm ảnh chi tiết nhưng tạo ra tokens có tính mạch lạc ngữ nghĩa yếu.
- Tối ưu hóa ngữ nghĩa (hiểu): SigLIP embeddings nhóm hình ảnh "mèo" gần tokens "mèo" nhưng không cho phép tái tạo tốt.

Show-o và Transfusion trả tiền cho việc này với thuế chất lượng có thể nhìn thấy theo một hướng. Janus-Pro đặt câu hỏi: tại sao lại yêu cầu một tokenizer khi các nhiệm vụ có nhu cầu khác nhau?

## Khái niệm

### Mã hóa hình ảnh tách rời

Kiến trúc của Janus-Pro tách biệt hai encoders:

- Hiểu con đường. Hình ảnh đầu vào → SigLIP-SO400m → thân → transformer MLP 2 lớp.
- Con đường thế hệ. Hình ảnh đầu vào (nếu điều kiện trên hình ảnh hiện có) → VQ tokenizer → token ID → transformer nội dung.
- Tạo đầu ra. Hình ảnh tokens dự đoán bởi transformer → VQ decoder → pixel.

Cơ thể transformer được chia sẻ. Mọi thứ ở thượng nguồn và hạ lưu của cơ thể đều có nhiệm vụ cụ thể.

Đầu vào được phân biệt theo định dạng prompt: thẻ `<understand>` định tuyến qua SigLIP; `<generate>` tuyến đường qua VQ. Hoặc định tuyến ngầm từ nhiệm vụ.

### Tại sao tính năng này hoạt động

Hiểu được loss có được SigLIP features, mà pretraining theo phong cách CLIP đã điều chỉnh để có sự tương đồng về ngữ nghĩa. Nhận thức của model benchmarks cải thiện so với Show-o / Truyền máu vì features đầu vào tốt hơn cho nhiệm vụ.

Thế hệ loss nhận được VQ tokens, mà một tokenizer đã điều chỉnh để tái tạo. Chất lượng hình ảnh được cải thiện so với Show-o vì mã VQ soạn trở lại pixel một cách rõ ràng.

Cơ thể transformer được chia sẻ nhìn thấy hai phân phối đầu vào (SigLIP và VQ) và học cách làm việc với cả hai. Tuyên bố: đủ dữ liệu + đủ parameters, cơ thể hấp thụ chuyển mạch.

### Mở rộng quy mô dữ liệu - Janus vs Janus-Pro

Janus (bản gốc, arXiv 2410.13848) đã giới thiệu việc tách rời nhưng ở quy mô nhỏ (tham số 1,3 tỷ tham số, dữ liệu hạn chế). Janus-Pro (arXiv 2501.17811) chia tỷ lệ:

- 7B tham số (so với 1,3B).
- 90 triệu cặp hình ảnh-văn bản cho giai đoạn 1 (alignment) tăng từ 72 triệu.
- 72M cho giai đoạn 2 (thống nhất) tăng từ 26M.
- Đã thêm 200k mẫu hướng dẫn tạo hình ảnh cho giai đoạn 3.

Kết quả: Janus-Pro-7B phù hợp với LLaVA trên MMMU (60.3 so với ~58) và đánh bại DALL-E 3 trên GenEval (0.80 so với 0.67). Một model mở, cạnh tranh ở cả hai phía của phổ thống nhất.

### JanusFlow — biến thể luồng chỉnh lưu

JanusFlow (arXiv 2411.07975) hoán đổi đường dẫn tạo VQ cho đường dẫn tạo dòng chỉnh lưu (liên tục). Sự phân tách trở thành SigLIP-for-understanding + rectified-flow-for-generation. Trần nhà chất lượng nâng cao hơn nữa. Kiến trúc vẫn tách rời encoders chia sẻ.

### Công việc của cơ thể được chia sẻ

Phần thân transformer processes một trình tự thống nhất nhưng có hai phân phối đầu vào. Công việc của nó là:

- Để hiểu: sử dụng SigLIP features + văn bản tokens → tự phát ra văn bản.
- Đối với thế hệ: sử dụng văn bản tokens + (hình ảnh tùy chọn VQ tokens) → phát ra VQ hình ảnh tokens tự hồi quy.

Cơ thể không có trọng lượng cụ thể theo phương thức trên mỗi khối. Đó là transformer kiểu văn bản mà bạn mong đợi tìm thấy bên trong Qwen hoặc Llama, cộng với hai bộ điều hợp đầu vào.

Điều thú vị là điều này có nghĩa là cơ thể của Janus-Pro có thể được khởi tạo từ một pretrained LLM. Janus-Pro khởi tạo từ DeepSeek-MoE-7B. Sự lựa chọn đó rất quan trọng: LLM đóng góp khả năng suy luận mà models thống nhất thuần túy từ đầu phải vật lộn để đạt được.

### So với InternVL-U

InternVL-U (Bài 12.10) là phần tiếp theo năm 2026. Nó kết hợp:

- pretraining đa phương thức gốc (đường trục InternVL3).
- Định tuyến encoder tách rời (SigLIP vào, VQ + đầu khuếch tán ra).
- Hiểu biết thống nhất + tạo + chỉnh sửa.

InternVL-U đưa lựa chọn kiến trúc của Janus-Pro vào một framework lớn hơn. Ý tưởng encoder tách rời hiện là mặc định cho models thống nhất trên quy mô lớn.

### Hạn chế

Tách rời encoders làm tăng thêm sự phức tạp về kiến trúc. Hai tokenizers để huấn luyện, hai đường dẫn đầu vào để duy trì, hai bộ chế độ lỗi. Đối với các sản phẩm không cần phát điện, Janus-Pro được thiết kế quá mức - hãy chọn một model hiểu biết về dòng LLaVA.

Đối với các sản phẩm không cần hiểu biết, Janus-Pro đủ tiêu chuẩn - hãy chọn model khuếch tán ổn định 3 / Flux.

Đối với các sản phẩm cần cả hai, Janus-Pro hiện là kiến trúc mở tham chiếu.

## Ứng dụng

`code/main.py` mô phỏng định tuyến Janus-Pro:

- Hai encoders giả: SigLIP-like (tạo ra vectors ngữ nghĩa 256 mờ) và VQ-like (tạo ra mã số nguyên).
- Một bộ định tuyến prompt chọn encoder dựa trên thẻ tác vụ.
- Một cơ thể được chia sẻ (thay thế) processes token các trình tự bất kể encoder nào tạo ra chúng.
- Chuyển đổi từ stage 1 (alignment) sang stage 3 (giai điệu hướng dẫn) có trọng số sample lịch trình.

In các đường dẫn định tuyến cho 3 ví dụ: QA hình ảnh, T2I, chỉnh sửa hình ảnh.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-decoupled-encoder-picker.md`. Với một sản phẩm muốn tạo ra + hiểu biết thống nhất ở chất lượng biên giới, nó chọn Janus-Pro, JanusFlow hoặc InternVL-U với đề xuất quy mô dữ liệu cụ thể.

## Bài tập

1. Janus-Pro-7B đánh bại DALL-E 3 trên GenEval. Giải thích lý do tại sao model mở 7B có thể phù hợp với model độc quyền biên giới về thế hệ nhưng không phù hợp với sự hiểu biết.

2. Triển khai chức năng bộ định tuyến: cho prompt văn bản, phân loại là `understand` hoặc `generate`. Làm thế nào để bạn xử lý các prompts mơ hồ như "mô tả và sau đó phác thảo"?

3. JanusFlow thay thế đường dẫn VQ bằng luồng chỉnh lưu. Cơ thể transformer bây giờ tạo ra những gì và những thay đổi nào trong loss?

4. Đề xuất nhiệm vụ thứ tư mà kiến trúc Janus-Pro có thể xử lý với một encoder tách rời nữa. Ví dụ: phân đoạn hình ảnh (kiểu DINO), độ sâu (kiểu MiDaS).

5. Đọc Janus-Pro Phần 4.2 về mở rộng dữ liệu. Giai đoạn dữ liệu nào đóng góp nhiều nhất vào việc tăng chất lượng T2I so với Janus?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Mã hóa tách rời | "Hai encoders trực quan" | Riêng biệt tokenizer hoặc encoder theo hướng: ngữ nghĩa để hiểu, tái tạo để tạo ra |
| Cơ thể chung | "Một transformer" | Một transformer processes đầu ra của một trong hai encoder; Không có trọng số cụ thể theo phương thức |
| SigLIP để hiểu | "features ngữ nghĩa" | Tháp tầm nhìn gia đình CLIP cung cấp features khái niệm phong phú nhưng tái tạo kém |
| VQ cho thế hệ | "Mã tái tạo" | tokens lượng tử hóa Vector giải mã rõ ràng trở lại pixel |
| Dòng chảy JanusFlow | "Biến thể dòng chảy chỉnh lưu" | Janus-Pro với đầu phát điện khớp dòng liên tục thay vì VQ |
| Thẻ định tuyến | "Thẻ nhiệm vụ" | Prompt điểm đánh dấu (`<understand>` / `<generate>`) chọn encoder đầu vào |

## Đọc thêm

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
