# Hiểu tài liệu và sơ đồ

> Tài liệu không phải là ảnh. PDF, bài báo khoa học, hóa đơn hoặc biểu mẫu viết tay có bố cục, bảng, sơ đồ, chú thích, tiêu đề và cấu trúc ngữ nghĩa mà sự hiểu biết bằng hình ảnh đơn giản không thể nắm bắt được. Tiền VLM stack là một pipeline: Tesseract OCR + LayoutLMv3 + phỏng đoán trích xuất bảng. Làn sóng VLM đã thay thế bằng các models không có OCR - Donut (2022), Nougat (2023), DocLLM (2023) - phát ra đánh dấu có cấu trúc trực tiếp. Đến năm 2026, biên giới chỉ là "cung cấp hình ảnh trang cho Claude Opus 4.7 ở 2576px gốc" và đầu ra đánh dấu có cấu trúc được cung cấp miễn phí. Bài học này đọc vòng cung ba thời đại của tài liệu AI.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, layout-aware document parser skeleton)
**Kiến thức tiên quyết:** Giai đoạn 12 · 05 (LLaVA), Giai đoạn 5 (NLP)
**Thời lượng:** ~180 phút

## Mục tiêu học tập

- Giải thích ba thời đại của AI tài liệu: OCR pipeline, không có OCR, VLM bản địa.
- Mô tả ba luồng đầu vào của LayoutLMv3: văn bản, bố cục (bbox), bản vá hình ảnh, với mặt nạ thống nhất.
- So sánh Donut (không có OCR, đánh dấu hình ảnh →), Nougat (bài báo khoa học → LaTeX), DocLLM (tổng quát nhận biết bố cục), PaliGemma 2 (VLM-native).
- Chọn một model tài liệu cho một nhiệm vụ mới (hóa đơn, bài báo khoa học, biểu mẫu viết tay, biên lai tiếng Trung).

## Vấn đề

"Hiểu PDF này" có vẻ khó khăn. Thông tin nằm trong:

- Nội dung văn bản (90% tín hiệu).
- Bố cục (đầu trang, chú thích, thanh bên, định dạng hai cột).
- Bảng (hàng, cột merged ô).
- Số liệu và sơ đồ.
- Chú thích viết tay.
- Phông chữ và kiểu chữ (tiêu đề so với nội dung).

Raw OCR kết xuất văn bản và mất rest. Một hệ thống quan tâm đến hóa đơn cần biết "Tổng cộng: 1.245 đô la" đến từ dưới cùng bên phải, không phải từ chú thích.

## Khái niệm

### Kỷ nguyên 1 - OCR pipeline (trước năm 2021)

stack cổ điển:

1. PDF → hình ảnh trên mỗi trang.
2. Tesseract (hoặc OCR thương mại) trích xuất văn bản với các hộp giới hạn cho mỗi từ.
3. Trình phân tích bố cục xác định các khối (tiêu đề, bảng, đoạn văn).
4. Bộ nhận dạng cấu trúc bảng phân tích cú pháp bảng.
5. Quy tắc miền + trường trích xuất biểu thức chính quy.

Hoạt động cho văn bản in sạch. Ngắt chữ viết tay, quét lệch, bảng phức tạp, scripts không phải tiếng Anh. Mọi chế độ lỗi đều yêu cầu một đường dẫn ngoại lệ tùy chỉnh.

### TrOCR (2021)

TrOCR (Li và cộng sự, arXiv: 2109.10282) đã thay thế CNN-CTC cổ điển của Tesseract bằng một decoder transformer encoder được huấn luyện trên hình ảnh tổng hợp + văn bản thực. Chiến thắng sạch sẽ trên văn bản viết tay và đa ngôn ngữ. Vẫn là một pipeline (máy dò sau đó là TrOCR sau đó bố trí), nhưng bước OCR đã được cải thiện đáng kể.

### Kỷ nguyên 2 - Không OCR (2022-2023)

models không có OCR đầu tiên cho biết: bỏ qua hoàn toàn phát hiện, ánh xạ trực tiếp các pixel hình ảnh đến đầu ra có cấu trúc.

Bánh rán (Kim và cộng sự, arXiv: 2111.15664):
- Encoder-decoder transformer, encoder là Swin-B.
- Đầu ra được JSON để hiểu biểu mẫu, đánh dấu để tóm tắt hoặc bất kỳ schema cụ thể nào của nhiệm vụ.
- Không OCR, không bố cục, không phát hiện.

Nougat (Blecher và cộng sự, arXiv: 2308.13418):
- Được huấn luyện đặc biệt về các bài báo khoa học.
- Đầu ra là LaTeX / markdown.
- Xử lý các phương trình, bố cục nhiều cột, hình.
- Các model mọi arXiv-parser gọi.

Đây là những chuyên gia, không phải những người tổng quát. Bánh rán trên một bài báo khoa học thất bại; Nougat trên hóa đơn không thành công.

### Bố cụcLMv3 (2022)

Một bản nhạc khác. LayoutLMv3 (Huang và cộng sự, arXiv:2204.08387) giữ OCR nhưng thêm cách hiểu bố cục:

- Ba luồng đầu vào: OCR tokens văn bản, hộp giới hạn 2D trên mỗi token, bản vá hình ảnh.
- Mặt nạ training mục tiêu trên cả ba phương thức (văn bản mặt nạ, bản vá mặt nạ, bố cục mặt nạ).
- Hạ nguồn: phân loại, trích xuất thực thể, bảng QA.

LayoutLMv3 là đỉnh cao của sự hiểu biết về tài liệu dựa trên OCR. Mạnh về biểu mẫu và hóa đơn. Yêu cầu OCR ngược dòng. VLM accuracy trước tốt nhất về benchmarks tài liệu tiêu chuẩn.

### Tài liệu LLM (2023)

DocLLM (Wang và cộng sự, arXiv: 2401.00908) là anh em tổng quát của LayoutLM. Tạo câu trả lời dạng tự do có điều kiện tokens bố cục. Tốt hơn cho QA trên tài liệu; vẫn phụ thuộc vào đầu vào OCR.

### Kỷ nguyên 3 - VLM bản địa (2024+)

Năm 2024 VLMs trở nên đủ tốt để thay thế hoàn toàn pipeline. Cung cấp hình ảnh toàn trang ở độ phân giải cao cho VLM, đặt câu hỏi, nhận câu trả lời.

- LLaVA-NeXT 336-tile AnyRes hoạt động cho các tài liệu nhỏ.
- Độ phân giải động Qwen2.5-VL xử lý 2048+ pixel nguyên bản.
- Claude Opus 4.7 hỗ trợ tài liệu 2576px.
- PaliGemma 2 (Tháng Tư 2025) huấn luyện dành riêng cho tài liệu + chữ viết tay.

Khoảng cách giữa VLM-bản địa và OCR-pipeline nhanh chóng thu hẹp. Đến năm 2026, người bản địa VLM giành chiến thắng:

- Văn bản cảnh (viết tay + in, scripts hỗn hợp).
- Bảng phức tạp với các ô merged.
- Phương trình toán học được nhúng trong văn bản.
- Số liệu có chú thích văn bản.

OCR pipelines vẫn giành chiến thắng trên:

- Khối lượng công việc quét thuần túy ở quy mô lớn, nơi độ trễ trên mỗi trang rất quan trọng.
- Pipeline độ tin cậy (thất bại xác định so với ảo giác VLM).
- Môi trường được quy định yêu cầu đầu ra OCR có thể kiểm toán.

### Biên giới Claude 4.7 / GPT-5

Ở đầu vào gốc 2576 pixel, biên giới VLMs thực hiện hiểu tài liệu ở accuracy gần con người. Những con số benchmark từ đầu năm 2026:

- DocVQA: Claude 4.7 ~95.1, PaliGemma 2 ~88.4, Nougat ~77.3, LayoutLMv3 ~83 có đường ống.
- Biểu đồQA: Claude 4,7 ~92,2, GPT-4V ~78.
- VisualMRC: Claude 4.7 ~94.

Khoảng cách model khép kín chủ yếu là độ phân giải và tỷ lệ LLM cơ sở. Mở models tại 7B kém vài điểm nhưng bắt kịp.

### Phương trình toán học và đầu ra LaTeX

Các bài báo khoa học cần đầu ra LaTeX chính xác cho các phương trình. Nougat đã được huấn luyện về điều này. VLMs được huấn luyện với các mục tiêu LaTeX (Qwen2.5-VL-Math, dẫn xuất Nougat) tạo ra LaTeX có thể sử dụng được. Nếu không có training LaTeX rõ ràng, VLMs tạo ra các bản phiên âm dễ đọc nhưng không chính xác.

Đối với bài báo khoa học pipelines vào năm 2026: chuỗi Nougat trên PDF, sau đó là VLM trên các trang khó.

### Chữ viết tay

Vẫn là nhiệm vụ phụ khó nhất. In hỗn hợp + viết tay (ghi chú của bác sĩ, biểu mẫu điền) là nơi OCR pipelines vẫn đánh bại VLMs về chi phí. Các VLMs chỉ viết tay đang được cải thiện (Claude 4.7, PaliGemma 2).

### Công thức năm 2026

Đối với dự án AI tài liệu mới:

- Hóa đơn in thuần túy trên quy mô lớn: Quy tắc LayoutLMv3+, tiết kiệm chi phí.
- Tài liệu hỗn hợp (khoa học + viết tay + biểu mẫu): VLM-bản địa (PaliGemma 2 hoặc Qwen2.5-VL).
- Nhập arXiv đầy đủ: Nougat cho toán học, VLM cho số liệu.
- Quy định: Trình xác thực OCR pipeline + VLM để kiểm tra chéo.

## Ứng dụng

`code/main.py`:

- Một tokenizer nhận biết bố cục đồ chơi: các cặp (văn bản, bbox) đã cho, tạo ra đầu vào kiểu LayoutLMv3.
- Trình tạo schema tác vụ kiểu bánh rán: JSON mẫu cho biểu mẫu.
- So sánh ngân sách token trên mỗi trang trên OCR-pipeline, Donut, Nougat và VLM-native.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-document-ai-stack-picker.md`. Được đưa ra một dự án AI tài liệu (miền, quy mô, chất lượng, quy định), lựa chọn giữa OCR pipeline, chuyên gia không có OCR và VLM gốc.

## Bài tập

1. Dự án của bạn là 10 triệu hóa đơn mỗi ngày. stack nào giảm thiểu chi phí mỗi trang mà không làm mất accuracy?

2. Tại sao LayoutLMv3 hoạt động tốt hơn CLIP-VLMs thuần túy trên biểu mẫu QA nhưng hoạt động kém hơn ở văn bản cảnh? Luồng bbox từ bỏ điều gì?

3. Nougat tạo ra LaTeX. Đề xuất một trường hợp thử nghiệm trong đó đầu ra gốc VLM đánh bại Nougat về độ trung thực của LaTeX và một trường hợp mà Nougat giành chiến thắng.

4. Đọc bài báo PaliGemma 2 (Google, 2024). Bổ sung dữ liệu training quan trọng đã nâng tài liệu accuracy so với PaliGemma 1 là gì?

5. Thiết kế kết hợp an toàn theo quy định: OCR pipeline là chính VLM là kiểm tra chéo thứ cấp. Làm thế nào để bạn giải quyết bất đồng?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| OCR pipeline | "Kiểu Tesseract" | stack theo giai đoạn: phát hiện -> OCR -> bố cục -> quy tắc; xác định, mong manh |
| Không OCR | "Kiểu bánh rán" | transformer hình ảnh thành đầu ra bỏ qua OCR rõ ràng; model đơn |
| Nhận biết bố cục | "Bố cụcLM" | Đầu vào bao gồm tọa độ bbox trên mỗi token; Mặt nạ thống nhất trên các phương thức |
| VLM bản địa | "Biên giới VLM" | Cung cấp hình ảnh trang trực tiếp vào Claude/GPT/Qwen VLM ở độ phân giải cao; Không pipeline |
| Tài liệuVQA | "Bác sĩ benchmark" | Tài liệu tiêu chuẩn VQA; Điểm được trích dẫn nhiều nhất |
| Đầu ra đánh dấu | "LaTeX / MD" | Định dạng đầu ra có cấu trúc thay vì văn bản dạng tự do; Cho phép tự động hóa hạ nguồn |

## Đọc thêm

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
