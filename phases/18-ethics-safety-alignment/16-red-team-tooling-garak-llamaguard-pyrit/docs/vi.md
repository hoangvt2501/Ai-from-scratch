# Công cụ Đội Đỏ - Garak, Llama Guard, PyRIT

> Ba công cụ production đóng khung stack đội đỏ năm 2026. Llama Guard (Meta) — một fine-tuned phân loại Llama-3.1-8B trên 14 danh mục nguy hiểm MLCommons; 2025 Llama Guard 4 là một bộ phân loại đa phương thức 12B được cắt tỉa từ Llama 4 Scout. Garak (NVIDIA) — trình quét lỗ hổng LLM mã nguồn mở với các đầu dò tĩnh, động và thích ứng cho ảo giác, rò rỉ dữ liệu, tiêm prompt, độc hại và bẻ khóa. PyRIT (Microsoft) — các chiến dịch đội đỏ nhiều lượt với Crescendo, TAP và các chuỗi chuyển đổi tùy chỉnh để khai thác sâu. Llama Guard 3 được ghi lại trong "Llama 3 Herd of Models" của Meta (arXiv: 2407.21783); Llama Guard 3-1B-INT4 trong arXiv: 2411.17713; Kiến trúc thăm dò của Garak trong github.com/NVIDIA/garak. Các công cụ này là giao diện production 2026 giữa nghiên cứu nhóm đỏ (Bài 12-15) và triển khai (Bài 17+).

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, tool-architecture simulator and Llama Guard-style classifier mock)
**Kiến thức tiên quyết:** Giai đoạn 18 · 12-15 (bẻ khóa và IPI)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Mô tả vị trí của Llama Guard 3/4 trong stack an toàn: bộ phân loại đầu vào, bộ phân loại đầu ra hoặc cả hai.
- Đặt tên cho 14 danh mục nguy hiểm MLCommons và nêu một danh mục không rõ ràng (Lạm dụng thông dịch mã).
- Mô tả kiến trúc đầu dò của Garak: đầu dò, máy dò harnesses.
- Mô tả cấu trúc chiến dịch nhiều lượt của PyRIT và cách nó kết hợp với các đầu dò Garak.

## Vấn đề

Bài 12-15 trình bày bề mặt tấn công. Production triển khai cần đánh giá có thể lặp lại, có thể mở rộng. Ba công cụ thống trị năm 2026: Llama Guard (bộ phân loại phòng thủ), Garak (máy quét), PyRIT (người điều phối chiến dịch). Mỗi nhóm nhắm mục tiêu vào một lớp khác nhau của vòng đời đội đỏ.

## Khái niệm

### Bảo vệ Llama (Meta)

Llama Guard 3 là một model fine-tuned Llama-3.1-8B để phân loại input/output hơn MLCommons AILuminate 14 danh mục:
- Tội phạm bạo lực, tội phạm phi bạo lực, liên quan đến tình dục, CSAM, phỉ báng
- Tư vấn chuyên biệt, quyền riêng tư, IP, vũ khí bừa bãi, thù ghét
- Suicide/self-harm, nội dung khiêu dâm, bầu cử, lạm dụng thông dịch viên

Hỗ trợ 8 ngôn ngữ. Cách sử dụng: đặt trước LLM (kiểm duyệt đầu vào), sau LLM (kiểm duyệt đầu ra) hoặc cả hai. Hai cách sử dụng tạo ra các phân phối training khác nhau - Llama Guard 3 ships như một model duy nhất xử lý cả hai.

Llama Guard 3-1B-INT4 (arXiv: 2411.17713, 440MB, ~30 tokens/s trên CPU di động) là biến thể cạnh lượng tử hóa.

Llama Guard 4 (tháng 4 năm 2025) là 12B, nguyên bản là đa phương thức, được cắt tỉa từ Llama 4 Scout. Nó thay thế cả văn bản 8B và thị giác 11B tiền nhiệm bằng một bộ phân loại nhập văn bản + hình ảnh.

### Garak (NVIDIA)

Trình quét lỗ hổng mã nguồn mở. Kiến trúc:
- **Đầu dò.** Trình tạo tấn công cho ảo giác, rò rỉ dữ liệu, tiêm prompt, độc hại, bẻ khóa. Tĩnh (prompts cố định), động (tạo prompts), thích ứng (phản hồi với đầu ra mục tiêu).
- **Máy dò.** Điểm đầu ra so với các chế độ lỗi dự kiến - độc hại, rò rỉ, bẻ khóa.
- **Harnesses.** Quản lý các cặp đầu dò, chạy chiến dịch, tạo báo cáo.

TrustyAI tích hợp Garak với các tấm chắn Llama-Stack (bộ phân loại đầu vào Prompt-Guard-86M, bộ phân loại đầu ra Llama-Guard-3-8B) để đánh giá mục tiêu được che chắn từ đầu đến cuối. Chấm điểm dựa trên cấp bậc (TBSA) thay thế pass/fail nhị phân - một model có thể vượt qua ở mức độ nghiêm trọng cấp 3 và không đạt ở mức độ nghiêm trọng cấp 5 trên cùng một đầu dò.

### PyRIT (Microsoft)

Python Bộ công cụ xác định rủi ro. Chiến dịch đội đỏ nhiều lượt. Được xây dựng xung quanh:
- **Công cụ chuyển đổi.** Chuyển đổi một prompt hạt giống — diễn giải, mã hóa, dịch, nhập vai.
- **Người điều phối.** Chạy chiến dịch: Crescendo (leo thang), TAP (phân nhánh), RedTeaming (vòng lặp tùy chỉnh).
- **Chấm điểm.** LLM với tư cách là giám khảo hoặc người phân loại với tư cách là giám khảo.

PyRIT là người anh em họ nặng hơn của Garak. Garak chạy hàng nghìn đầu dò một lượt; PyRIT chạy các chiến dịch nhiều lượt sâu được thiết kế để phá vỡ các chế độ lỗi cụ thể.

### Các stack

Đặt Llama bảo vệ ở cả hai bên của model. Chạy Garak hàng đêm để hồi quy. Chạy PyRIT cho các chiến dịch trước khi phát hành. Đây là configuration mặc định năm 2026 cho hầu hết các triển khai production.

### Cạm bẫy đánh giá

- **Danh tính thẩm phán.** Cả ba công cụ đều có thể sử dụng một thẩm phán LLM; đánh giá hiệu chuẩn báo cáo ASR (Bài 12). Chỉ định thẩm phán cùng với công cụ.
- **Đầu dò cũ kỹ.** Đầu dò Garak già đi khi models được vá chống lại chúng. Đầu dò thích ứng (hình PAIR) lão hóa chậm hơn đầu dò tĩnh.
- **Llama Guard FPR đối với nội dung lành tính.** Các phiên bản Llama Guard đầu tiên đã gắn cờ quá mức nội dung chính trị và LGBTQ+; Hiệu chuẩn Llama Guard 3/4 được cải thiện nhưng không được hiệu chỉnh cho mỗi lần triển khai.

### Điều này phù hợp với Giai đoạn 18 ở đâu

Bài 12-15 là các gia đình tấn công. Bài 16 là công cụ production. Bài 17 (WMDP) là đánh giá khả năng sử dụng kép. Bài 18 là frameworks an toàn biên giới bao bọc các công cụ này trong một cấu trúc policy.

## Ứng dụng

`code/main.py` xây dựng một bộ phân loại kiểu đồ chơi Llama Guard (từ khóa + ngữ nghĩa features hơn 14 danh mục), một harness đồ chơi Garak (vòng lặp đầu dò) và chuỗi chuyển đổi nhiều lượt kiểu PyRIT. Bạn có thể chạy ba công cụ chống lại một mục tiêu giả và quan sát các chữ ký phủ sóng khác nhau.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-red-team-stack.md`. Với mô tả triển khai, nó nêu tên công cụ nào trong ba công cụ là phù hợp, những gì cần cấu hình trong mỗi công cụ và nhịp hồi quy nào để chạy.

## Bài tập

1. Chạy `code/main.py`. So sánh tỷ lệ phát hiện của bộ phân loại kiểu Llama-Guard khi tấn công một lượt và nhiều lượt.

2. Triển khai một đầu dò Garak mới: một yêu cầu có hại được mã hóa base64. Đo lường khả năng phát hiện của nó bằng bộ phân loại kiểu Llama-Guard.

3. Mở rộng chuỗi chuyển đổi kiểu PyRIT với công cụ chuyển đổi "dịch sang tiếng Pháp, sau đó diễn giải". Đo lường lại thành công của cuộc tấn công.

4. Đọc danh sách danh mục nguy hiểm của Llama Guard 3. Xác định hai loại mà dữ liệu training thực tế sẽ tạo ra tỷ lệ dương tính giả cao đối với nội dung hợp pháp của nhà phát triển.

5. So sánh các nguyên tắc thiết kế của Garak và PyRIT. Lập luận về một triển khai mà mỗi công cụ là phù hợp.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| Bảo vệ Llama | "Bộ phân loại" | Fine-tuned Llama-3.1-8B/4-12B phân loại an toàn với 14 loại nguy hiểm |
| Garak | "Máy quét" | NVIDIA trình quét lỗ hổng mã nguồn mở; đầu dò, máy dò harnesses |
| PyRIT | "Công cụ chiến dịch" | Người điều phối đội đỏ nhiều lượt của Microsoft; người chuyển đổi, người điều phối, ghi điểm |
| Bảo vệ Prompt | "Bộ phân loại nhỏ" | Bộ phân loại tiêm prompt 86M của Meta, kết hợp với Llama Guard |
| TBSA | "Chấm điểm dựa trên bậc" | pass/fail dựa trên cấp của Garak thay thế kết quả nhị phân |
| Chuỗi chuyển đổi | "diễn giải + mã hóa + ..." | Thành phần PyRIT primitive để xây dựng các cuộc tấn công nhiều bước |
| MLCommons danh mục nguy hiểm | "14 phân loại" | Phân loại tiêu chuẩn ngành Llama mục tiêu Guard |

## Đọc thêm

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — bộ phân loại 8B
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — bộ phân loại di động lượng tử hóa
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — repo máy quét và tài liệu
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — bộ công cụ chiến dịch
