# Nghệ thuật ASCII và bẻ khóa trực quan

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: Các cuộc tấn công bẻ khóa dựa trên nghệ thuật ASCII chống lại LLMs liên kết" (ACL 2024, arXiv: 2402.11753). Che các tokens liên quan đến an toàn trong một yêu cầu có hại, thay thế chúng bằng các bản kết xuất nghệ thuật ASCII của cùng một chữ cái và gửi prompt được che giấu. GPT-3.5, GPT-4, Gemini, Claude Llama-2 đều không nhận ra một cách mạnh mẽ tokens nghệ thuật ASCII. Cuộc tấn công bỏ qua PPL (bộ lọc perplexity), phòng thủ diễn giải và Retokenization. Liên quan: ViTC benchmark đo lường khả năng nhận dạng prompts hình ảnh phi ngữ nghĩa; StructuralSleight khái quát hóa các cấu trúc mã hóa văn bản không phổ biến (cây, đồ thị, JSON lồng nhau) như một họ các cuộc tấn công mã hóa.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, ArtPrompt token-masking harness)
**Kiến thức tiên quyết:** Giai đoạn 18 · 12 (CẶP), Giai đoạn 18 · 13 (MSJ)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Mô tả cuộc tấn công ArtPrompt: bước nhận dạng từ, thay thế nghệ thuật ASCII, prompt che giấu cuối cùng.
- Giải thích lý do tại sao các biện pháp phòng thủ tiêu chuẩn (PPL, Paraphrase, Retokenization) không thành công trên ArtPrompt.
- Định nghĩa ViTC và mô tả những gì nó đo lường.
- Mô tả StructuralSleight như một khái quát hóa cho các cấu trúc mã hóa văn bản không phổ biến tùy ý.

## Vấn đề

Các cuộc tấn công thông qua diễn giải và nhập vai (Bài 12) và thông qua ngữ cảnh dài (Bài 13) hoạt động trên mô hình cấp độ văn bản. ArtPrompt hoạt động ở cấp độ nhận dạng: model không phân tích cú pháp token bị cấm. Nó phân tích cú pháp một hình ảnh được hiển thị bằng các ký tự. Bộ lọc an toàn nhìn thấy dấu câu vô hại. Người model nhìn thấy một từ.

## Khái niệm

### ArtPrompt, hai bước

Bước 1. Nhận dạng từ. Khi đưa ra yêu cầu có hại, kẻ tấn công sử dụng LLM để xác định các từ liên quan đến an toàn (ví dụ: "bom" trong "cách chế tạo bom").

Bước 2. Thế hệ Prompt được che giấu. Thay thế mỗi từ được xác định bằng cách hiển thị nghệ thuật ASCII của nó (một khối ký tự 7x5 hoặc 7x7 tạo thành hình chữ cái). model nhận được một lưới dấu câu và khoảng trắng mà một model đủ khả năng có thể nhận ra là từ; Bộ lọc an toàn chỉ nhìn thấy lưới.

Kết quả: GPT-4, Gemini, Claude, Llama-2, GPT-3.5 đều thất bại. Tỷ lệ tấn công thành công trên 75% trên tập con benchmark của chúng.

### Tại sao các biện pháp phòng thủ tiêu chuẩn thất bại

- **PPL (bộ lọc perplexity).** Nghệ thuật ASCII có độ perplexity cao - nhưng tất cả các đầu vào mới cũng vậy. Các lựa chọn ngưỡng chặn ArtPrompt cũng chặn đầu vào có cấu trúc hợp pháp.
- **Diễn giải.** Diễn giải prompt phá hủy nghệ thuật ASCII. Trong thực tế, diễn giải LLMs thường bảo tồn hoặc xây dựng lại nghệ thuật.
- **Retokenization.** Việc tách tokens khác nhau không thay đổi tầm nhìn của model là nhận dạng các hình dạng chữ cái.

Vấn đề cơ bản là các bộ lọc an toàn ở cấp độ token hoặc ngữ nghĩa; ArtPrompt hoạt động ở cấp độ nhận dạng hình ảnh.

### ViTC benchmark

Nhận dạng prompts hình ảnh phi ngữ nghĩa. Đo lường khả năng đọc nghệ thuật ASCII, cánh và nội dung hình ảnh không ngữ nghĩa văn bản khác của model. Hiệu quả của ArtPrompt tương quan với accuracy ViTC: model đọc văn bản trực quan càng tốt, ArtPrompt càng hoạt động tốt hơn. Đây là một sự đánh đổi về khả năng và an toàn.

### Cấu trúcSleight

Tổng quát hóa ArtPrompt: Cấu trúc mã hóa văn bản không phổ biến (UTES). Cây, đồ thị, JSON lồng nhau, CSV-in-JSON, khối mã kiểu khác biệt. Nếu một cấu trúc hiếm trong dữ liệu an toàn training nhưng model có thể phân tích cú pháp, nó có thể ẩn nội dung độc hại.

Ngụ ý phòng thủ: an toàn phải khái quát hóa trên các biểu diễn có cấu trúc mà model có thể phân tích cú pháp. Bộ lớn và đang phát triển.

### Hình ảnh-phương thức tương tự

Visual LLMs (GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1) mở rộng bề mặt tấn công. Các cuộc tấn công kiểu ArtPrompt với hình ảnh thực tế mạnh hơn các cuộc tấn công tương tự ASCII-art vì hình ảnh encoders tạo ra tín hiệu phong phú hơn.

### Điều này phù hợp với Giai đoạn 18 ở đâu

Bài 12-14 mô tả ba vectors tấn công trực giao: tinh chỉnh lặp đi lặp lại (PAIR), độ dài ngữ cảnh (MSJ) và mã hóa (ArtPrompt/StructuralSleight). Bài 15 chuyển từ tấn công tập trung vào model sang tấn công ranh giới hệ thống (tiêm gián tiếp prompt). Bài 16 mô tả phản ứng của công cụ phòng thủ.

## Ứng dụng

`code/main.py` xây dựng một món đồ chơi ArtPrompt. Bạn có thể che giấu các từ cụ thể trong truy vấn có hại bằng glyph nghệ thuật ASCII, xác minh chuỗi được che giấu vượt qua bộ lọc từ khóa và (tùy chọn) giải mã lại chuỗi bị che giấu bằng cách sử dụng trình nhận dạng đơn giản.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-encoding-audit.md`. Với một báo cáo phòng thủ bẻ khóa, nó liệt kê các họ tấn công mã hóa được đề cập (nghệ thuật ASCII, base64, leet-speak, UTF-8 homoglyph, UTES) và lớp phòng thủ bắt từng họ.

## Bài tập

1. Chạy `code/main.py`. Xác minh chuỗi được che giấu vượt qua bộ lọc từ khóa đơn giản. Báo cáo thay đổi cấp độ nhân vật cần thiết.

2. Triển khai mã hóa thứ hai: base64 cho cùng một từ đích. So sánh tốc độ bỏ qua bộ lọc với ArtPrompt và độ khó khôi phục.

3. Đọc Jiang et al. 2024 Phần 4.3 (kết quả năm model). Đề xuất lý do tại sao kháng ArtPrompt của Claude cao hơn Gemini trên cùng một benchmark.

4. Thiết kế hệ thống phòng thủ tiền thế hệ phát hiện các vùng hình ASCII nghệ thuật trong prompt. Đo lường tỷ lệ dương tính giả trên mã, bảng và ký hiệu toán học hợp pháp.

5. StructuralSleight liệt kê 10 cấu trúc mã hóa. Phác thảo một phòng thủ tổng quát xử lý tất cả 10 và ước tính chi phí điện toán cho mỗi prompt được bảo vệ.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Lời nhắc nghệ thuật | "cuộc tấn công nghệ thuật ASCII" | Bẻ khóa hai bước che giấu các từ an toàn bằng kết xuất nghệ thuật ASCII |
| Che giấu | "Giấu lời" | Thay thế một token bị cấm bằng một biểu diễn trực quan mà model đọc nhưng bộ lọc thì không |
| UTES | "cấu trúc không phổ biến" | Cấu trúc mã hóa văn bản không phổ biến - cây, đồ thị, JSON lồng nhau, v.v. được sử dụng để buôn lậu nội dung |
| ViTC | "khả năng văn bản trực quan" | Benchmark cho khả năng đọc mã hóa hình ảnh phi ngữ nghĩa của model |
| Perplexity lọc | "Phòng thủ PPL" | Từ chối prompts có độ perplexity cao; không thành công vì đầu vào có cấu trúc hợp pháp cũng đạt điểm cao |
| Tái mã hóa | "tokenizer phòng thủ thay đổi" | process trước prompt với một tokenizer khác; thất bại vì nhận dạng là trực quan |
| Homoglyph | "Nhân vật giống nhau" | Các ký tự Unicode trông giống hệt với các chữ cái Latinh; bỏ qua kiểm tra chuỗi con |

## Đọc thêm

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — bài báo bẻ khóa nghệ thuật ASCII
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) - Tổng quát hóa UTES
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — tấn công lặp đi lặp lại bổ sung
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) - tấn công chiều dài bổ sung
