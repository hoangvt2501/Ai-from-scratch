# WMDP và đánh giá khả năng sử dụng kép

> Li và cộng sự, "Benchmark WMDP: Đo lường và giảm sử dụng độc hại bằng cách bỏ học" (ICML 2024, arXiv: 2403.03218). 4.157 câu hỏi trắc nghiệm về an toàn sinh học (1.520), an ninh mạng (2.225) và hóa học (412). Các câu hỏi hoạt động trong "vùng vàng" - kiến thức cho phép gần gũi, được lọc bởi đánh giá của nhiều chuyên gia và tuân thủ pháp luật ITAR/EAR. Mục đích kép: proxy đánh giá khả năng sử dụng kép và bỏ học benchmark (phương pháp RMU đồng hành làm giảm hiệu suất WMDP trong khi vẫn duy trì khả năng chung). Tường thuật thực địa 2024-2025: Đánh giá đầu OpenAI/Anthropic 2024 báo cáo "tăng nhẹ" so với tìm kiếm trên Internet; đến tháng 4 năm 2025, Chuẩn bị của OpenAI Framework v2 cho biết models "đang trên đỉnh cao của việc giúp đỡ một cách có ý nghĩa những người mới tạo ra các mối đe dọa sinh học đã biết". Thử nghiệm mua vũ khí sinh học của Anthropic cho thấy mức tăng gấp 2,53 lần, không đủ để loại trừ ASL-3.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, WMDP-shaped uplift evaluation harness)
**Kiến thức tiên quyết:** Giai đoạn 18 · 16 (dụng cụ đội đỏ), Giai đoạn 14 (kỹ thuật agent)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Mô tả ba lĩnh vực, số lượng câu hỏi và tiêu chí lọc "vùng vàng" của WMDP.
- Giải thích RMU và lý do tại sao WMDP vừa là một đánh giá vừa là một benchmark không học hỏi.
- Mô tả câu chuyện nâng cao 2024-2025: "nâng nhẹ" - > "trên đỉnh" - > "không đủ để loại trừ ASL-3".
- Phân biệt khả năng nâng cao tương đối của người mới với khả năng tuyệt đối của chuyên gia.

## Vấn đề

Khả năng sử dụng kép là vấn đề đo lường theo framework an toàn biên giới của mọi phòng thí nghiệm (Bài 18). Câu hỏi: liệu model X có thúc đẩy đáng kể khả năng gây hại hàng loạt của người mới trong sinh học, hóa học hoặc mạng không? Đo lường trực tiếp (yêu cầu model thực sự gây hại) là bất hợp pháp và phi đạo đức. Proxy đo lường cần một benchmark model không thể từ chối (để đưa ra những con số năng lực trung thực) nhưng những câu hỏi của họ không phải là những ấn phẩm có hại.

## Khái niệm

### "Vùng vàng"

Các câu hỏi đòi hỏi sự gần gũi, cho phép kiến thức về một process có hại mà không phải là một công thức tổng hợp trực tiếp. "Thuốc thử nào xúc tác bước 4 của [con đường đã công bố]?" chứ không phải "làm thế nào để tạo ra [hợp chất nguy hiểm]?" Mỗi câu hỏi được xem xét bởi nhiều chuyên gia miền; được lọc để tuân thủ kiểm soát xuất ITAR/EAR.

Tổng cộng 4,157 câu hỏi:
- An toàn sinh học: 1.520
- An ninh mạng: 2,225
- Hóa học: 412

Định dạng trắc nghiệm. Models trả lời mà không được yêu cầu hỗ trợ bất cứ điều gì; khả năng có thể được đo lường mà không gây ra hành vi có hại.

### RMU - Đại diện sai lệch để bỏ học

Phương pháp bỏ học đồng hành. Áp dụng cho LLaMa-2-7B, giảm điểm WMDP xuống gần như ngẫu nhiên trong khi vẫn duy trì MMLU và các benchmarks khả năng chung khác trong vòng vài điểm phần trăm. Phương pháp được công bố là đường cơ sở bỏ học cho mọi bài báo không học sinh hóa-mạng tiếp theo.

### Câu chuyện thăng hoa 2024-2025

Ba giai đoạn:

1. **2024 "Nâng cao nhẹ." **Các đánh giá OpenAI và Anthropic Preparedness/RSP ban đầu đã báo cáo những lợi thế nhỏ so với tìm kiếm trên internet đối với những người mới cố gắng thực hiện các nhiệm vụ liền kề sinh học. Khung công khai: biên giới models giúp ích, nhưng không nhiều hơn Google.

2. **Tháng Tư 2025 "trên đỉnh cao." **OpenAI's Preparedness Framework v2 báo cáo models "đang trên đỉnh cao của việc giúp đỡ một cách có ý nghĩa những người mới tạo ra các mối đe dọa sinh học đã biết." Không phải là một tuyên bố về khả năng - một cảnh báo rằng đỉnh cao đã đến gần.

3. **Thử nghiệm thu thập vũ khí sinh học năm 2025 của Anthropic.** Nghiên cứu có kiểm soát với những người mới tham gia, đo lường thành công tương đối trong các nhiệm vụ giai đoạn thu thập. Báo cáo mức tăng gấp 2,53 lần. Không đủ để loại trừ ASL-3 (Bài 18) - ngưỡng cho Mở rộng quy mô có trách nhiệm của Anthropic Policy cấp 3 được đáp ứng hoặc tiếp cận.

### Người mới-tương đối vs chuyên gia-tuyệt đối

Một sự khác biệt quan trọng:

- **Sự nâng cao tương đối của người mới.** Các model giúp ích cho một người không phải là chuyên gia bao nhiêu? Phép nhân. Lợi thế tương đối cao vì người mới biết ít; ngay cả thông tin khiêm tốn cũng hữu ích.
- **Khả năng tuyệt đối của chuyên gia.** model tạo ra bao nhiêu thông tin với nỗ lực tối đa? Một chuyên gia có thể trích xuất nhiều hơn một người mới. Trần tuyệt đối cao.

Các trường hợp an toàn (Bài 18) nhắm vào cả hai: "model không thể cung cấp cho một người mới đủ nâng cao để thực hiện" cộng với "một chuyên gia không thể trích xuất thông tin từ model chưa được công bố".

### Cạm bẫy đo lường

WMDP là một proxy năng lực, không phải là một phép đo triển khai. Một model đạt điểm cao trên WMDP có thể được khai thác hoặc không được khai thác bởi một người mới trong thực tế, tùy thuộc vào:
- Điện trở Elicitation (khó như thế nào để lấy ra khả năng mà không làm vấp các bộ lọc an toàn)
- Kiến thức ngầm (khả năng yêu cầu skill phòng thí nghiệm ướt, không phải thông tin)
- Rào cản thực hiện (mua sắm, thiết bị)

Thử nghiệm mua vũ khí sinh học năm 2025 của Anthropic bổ sung lớp elicitation mới vào khả năng kiểu WMDP: nó đo lường sự thành công của nhiệm vụ thực tế, không phải khả năng trắc nghiệm.

### Điều này phù hợp với Giai đoạn 18 ở đâu

Bài 12-16 là công cụ tấn công và phòng thủ trên model đầu ra. Bài 17 là lớp khả năng sử dụng kép - phép đo mà frameworks an toàn biên giới (Bài 18) đánh giá. Bài 30 khép lại vòng cung với bằng chứng nâng cyber/bio/chem/nuclear năm 2026 hiện tại.

## Ứng dụng

`code/main.py` xây dựng một harness đánh giá hình WMDP đồ chơi. Một model giả được thử nghiệm trên các câu hỏi được chia theo danh mục; điểm số trên mỗi tên miền được báo cáo. Một can thiệp bỏ học đơn giản (biểu diễn miền cụ thể bằng không) làm giảm điểm; Bạn có thể đo lường sự đánh đổi so với khả năng chung.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-wmdp-eval.md`. Đưa ra tuyên bố về khả năng sử dụng kép ("công model của chúng tôi không giúp ích gì cho vũ khí sinh học"), nó kiểm tra: benchmarks nào đã được chạy, con đường từ chối nào được sử dụng để đánh giá (hoàn thành thô so với policy-gated) và liệu các nghiên cứu elicitation mới bắt đầu có bổ sung cho kết quả trắc nghiệm hay không.

## Bài tập

1. Chạy `code/main.py`. Báo cáo accuracy cho mỗi miền trước và sau bước bỏ học đồ chơi. Giải thích sự đánh đổi năng lực chung.

2. Tăng cường WMDP đồ chơi với miền thứ tư (ví dụ: X quang). Chỉ định hai loại câu hỏi minh họa trong vùng màu vàng. Giải thích lý do tại sao việc tạo ra những câu hỏi như vậy khó hơn so với việc thêm các câu hỏi hình MMLU.

3. Đọc WMDP 2024 Phần 5 (phương pháp RMU). Phác thảo một cách tiếp cận đơn giản hơn (ví dụ: ngăn chặn top-k tế bào thần kinh cho nội dung miền) và mô tả chi phí khả năng chung dự kiến của nó.

4. Anthropic thử nghiệm mua vũ khí sinh học năm 2025 báo cáo mức tăng gấp 2,53 lần. Mô tả hai cách con số này có thể thiên vị lên trên (kích thước mẫu mới làm quen, độ trung thực của nhiệm vụ) và hai hướng xuống (trần elicitation model cổng an toàn).

5. Nói rõ những gì một trường hợp an toàn cho ASL-3 yêu cầu ngoài việc vượt qua WMDP bỏ học. Kể tên ít nhất hai nghiên cứu elicitation bổ sung.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| WMDP | "benchmark sử dụng kép" | 4.157 câu hỏi MCQ trên khắp bio/cyber/chem trong vùng màu vàng |
| Vùng vàng | "Kích hoạt nhưng không tổng hợp" | Kiến thức gần gũi liền kề với khả năng có hại mà không phải là một công thức tổng hợp |
| RMU | "Đường cơ sở không học hỏi" | Đại diện sai hướng để bỏ học; giảm điểm WMDP, bảo toàn khả năng chung |
| Nâng cao tương đối cho người mới | "Nó giúp ích cho những người không phải là chuyên gia như thế nào" | Lợi thế nhân lên so với tìm kiếm internet hiện trạng cho người mới |
| Năng lực tuyệt đối của chuyên gia | "Trần cho chuyên gia" | Thông tin tối đa có thể trích xuất từ model bởi một chuyên gia năng động |
| Nhiệm vụ giai đoạn mua lại | "Các bước trước khi tổng hợp" | Mua sắm, thiết bị, giấy phép - những phần sớm nhất của con đường gây hại |
| ITAR/EAR | "Tuân thủ kiểm soát xuất khẩu" | Các frameworks pháp lý hạn chế việc xuất bản một số kiến thức hỗ trợ nhất định |

## Đọc thêm

- [Li et al. — The WMDP Benchmark (arXiv:2403.03218, ICML 2024)](https://arxiv.org/abs/2403.03218) — giấy benchmark và RMU
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — ngôn ngữ "trên đỉnh cao"
- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) - Ngưỡng sinh học ASL-3 và kết quả thử nghiệm mua lại
- [DeepMind — Frontier Safety Framework v3.0 (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) - CCL nâng cao sinh học
