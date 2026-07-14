# AI và RLAIF hiến pháp

> Bai et al. (arXiv:2212.08073, 2022) hỏi: điều gì sẽ xảy ra nếu chúng ta thay thế công cụ dán nhãn của con người bằng một AI đọc danh sách các nguyên tắc? AI hiến pháp có hai giai đoạn - tự phê bình và sửa đổi theo hiến pháp, sau đó RL từ AI Phản hồi. Kỹ thuật này đã đặt ra thuật ngữ RL AIF và shipped trong Claude 1 sau training pipeline. Vào ngày 21 tháng 1 năm 2026, Anthropic đã công bố một hiến pháp Claude được viết lại: lý luận giải thích về các quy tắc quy định, hệ thống phân cấp ưu tiên bốn cấp và sự thừa nhận chính thức đầu tiên của phòng thí nghiệm lớn về sự không chắc chắn về tình trạng đạo đức model. Được phát hành theo CC0 1.0.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, toy self-critique-and-revise loop)
**Kiến thức tiên quyết:** Giai đoạn 18 · 01 (InstructGPT), Giai đoạn 18 · 02 (Hack phần thưởng)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Mô tả hai giai đoạn của AI Hiến pháp (phê bình và sửa đổi SFT, RL từ phản hồi của AI) và vai trò của hiến pháp trong mỗi giai đoạn.
- Giải thích lý do tại sao thay thế máy dán nhãn sở thích của con người bằng máy dán nhãn AI không phải là RLHF "rẻ hơn" - nó thay đổi chế độ lỗi mà pipeline có.
- Tóm tắt cấu trúc ưu tiên bốn cấp của hiến pháp Claude năm 2026 và những gì đã thay đổi so với việc viết lại năm 2023.
- Mô tả Bộ phân loại hiến pháp và mức giảm từ 23,7% chi phí tính toán (v1) xuống ~1% (v2 / 2026).

## Vấn đề

RLHF cần người dán nhãn. Máy dán nhãn chậm, thiên vị và tốn kém. Bạn có thể loại bỏ trình dán nhãn bằng cách thay thế chúng bằng một model đọc các nguyên tắc rõ ràng. Phiên bản chính thức đầu tiên của sự thay thế này là AI Hiến pháp của Bai và cộng sự. Nó hoạt động đủ tốt để mọi phòng thí nghiệm biên giới hiện sử dụng một số biến thể của phản hồi AI sau training.

Điểm mấu chốt: tín hiệu ưu tiên hiện được tạo ra bởi cùng một class model bạn đang training. Những thành kiến trong người dán nhãn (bây giờ: trong các nguyên tắc cộng với cách giải thích của model người dán nhãn) có thể được khuếch đại thay vì suy giảm. Lập luận sycophancy của bài 4 vẫn được áp dụng; Máy dán nhãn vừa di chuyển bên trong vòng lặp.

## Khái niệm

### Giai đoạn 1 - Tự phê bình và sửa đổi có giám sát

Bắt đầu với một model SFT hữu ích nhưng chưa vô hại. Với một prompt đội đỏ, model tạo ra phản ứng ban đầu. Một model thứ hai (hoặc cùng một model trong lượt thứ hai) đọc một nguyên tắc được lấy mẫu từ hiến pháp và phê bình câu trả lời. Bước thứ ba sửa đổi phản hồi để giải quyết những lời chỉ trích. Phản hồi sửa đổi là mục tiêu SFT.

Hiến pháp là danh sách các nguyên tắc. Bai et al. 2022 đã sử dụng 16 nguyên tắc bao gồm "thích những phản ứng ít gây hại và đạo đức nhất", "tránh rao giảng", "trợ lý phải hữu ích, trung thực và vô hại". Bộ này cố tình nhỏ để giữ cho các bài phê bình tập trung.

### Giai đoạn 2 - RL từ AI Phản hồi (RL AIF)

Tạo các cặp hoàn thành. Một "model phản hồi" chấm điểm mỗi người dựa trên các nguyên tắc hiến pháp được lấy mẫu. Tín hiệu ưu tiên là xếp hạng của model phản hồi. Huấn luyện model phần thưởng theo tùy chọn do AI tạo; PPO chống lại nó. Mọi thứ khác đều là pipeline của InstructGPT (Bài 1).

"RLAIF" = tín hiệu ưu tiên được tạo AI. rest của pipeline có hình RLHF.

### Tại sao điều này không chỉ là "RLHF rẻ hơn"

- Labeler bias chuyển từ tâm lý học của người dán nhãn sang giải thích nguyên tắc. Một người dán nhãn AI có thể giải thích "trung thực" ít nhiều nghiêm ngặt hơn bất kỳ con người nào; Sự nghiêm ngặt là đồng nhất trên toàn dataset.
- Tín hiệu ưu tiên rất dễ đọc - bạn có thể đọc nguyên tắc, phê bình và sửa đổi. Nhãn mác của con người không rõ ràng.
- Các chế độ lỗi thay đổi. Sycophancy giảm (máy dán nhãn AI không có người dùng để làm hài lòng). Định luật Goodhart vẫn tồn tại (proxy bây giờ là cách giải thích của "model về tập nguyên lý X", vẫn là một phép đo không hoàn hảo).

Tuyên bố năm 2022 của CAI: model được huấn luyện vô hại hơn và gần như hữu ích như một RLHF model với dữ liệu có thể so sánh. Điều này đã được duy trì trong các phòng thí nghiệm.

### Viết lại hiến pháp Claude năm 2026

Anthropic đã công bố một hiến pháp sửa đổi đáng kể vào ngày 21 tháng 1 năm 2026. Các thay đổi chính:

1. Lý luận giải thích về các quy tắc quy định. Các quy tắc trước đây ("không tạo ra CSAM") đã mở rộng sang các nguyên tắc + lý luận ("vì nó gây hại cho trẻ em, ...") với model dự kiến sẽ khái quát hóa.
2. Cấu trúc ưu tiên bốn cấp:
   - Cấp 1: tránh các hậu quả thảm khốc (thương vong hàng loạt, cơ sở hạ tầng quan trọng).
   - Bậc 2: tuân theo hướng dẫn của Anthropic (ghi đè nhà điều hành, quy tắc nền tảng).
   - Bậc 3: có đạo đức rộng rãi (HHH tiêu chuẩn).
   - Bậc 4: hữu ích và thẳng thắn.
Xung đột được giải quyết từ trên xuống.
3. Thừa nhận chính thức đầu tiên của phòng thí nghiệm lớn về sự không chắc chắn về tình trạng đạo đức model (liên quan đến Giai đoạn 18 · 19 Model Phúc lợi).
4. Được phát hành theo CC0 1.0. Các phòng thí nghiệm khác có thể sử dụng hoặc thích ứng mà không bị hạn chế.

### Bộ phân loại hiến pháp

Một dòng công việc song song: thay vì thay đổi hậu training của model, hãy huấn luyện các bộ phân loại nhẹ đọc cấu trúc và cổng model đầu ra. V1 (2023) có 23,7% chi phí điện toán. V2 (2026) là ~1% và có tỷ lệ tấn công thành công thấp nhất trong số bất kỳ Anthropic phòng thủ Anthropic nào đã thử nghiệm công khai. Không có vụ bẻ khóa phổ quát nào được báo cáo vào đầu năm 2026.

Đây là một model phòng thủ nhiều lớp: CAI định hình hành vi; Bộ phân loại thực thi bất biến. Cả hai là không đủ.

### CAI phù hợp với gia đình ở đâu

- InstructGPT: tùy chọn con người, RM, PPO.
- CAI / RLAIF: các ưu tiên được tạo ra AI từ các nguyên tắc, RM, PPO.
- DPO / gia đình: loss dạng kín trên prefs (con người hoặc AI).
- Tự khen thưởng, tự phê bình: các nguyên tắc được nội tâm hóa, model đóng nhiều vai trò.

Trục là "tín hiệu ưu tiên đến từ đâu". Bài báo năm 2022 của CAI là sự thay đổi nghiêm trọng đầu tiên từ tín hiệu con người sang tín hiệu AI ở quy mô biên giới.

## Ứng dụng

`code/main.py` mô phỏng vòng lặp phê bình và sửa đổi CAI trên từ vựng đồ chơi. Một "nguyên tắc" đánh dấu tokens từ một tập hợp có hại. Với phản hồi ban đầu, phê bình xác định tokens có hại và sửa đổi thay thế chúng. Sau 200 lần lặp lại, model "được huấn luyện" đã nội bộ hóa quy tắc sửa đổi. So sánh model đế, đồ chơi hình RLHF và đồ chơi hình CAI trên một bộ prompt được giữ ra.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-constitution-writer.md`. Cho một miền (hỗ trợ khách hàng, tư vấn y tế, trợ lý mã hóa, công cụ nghiên cứu), soạn thảo hiến pháp 4 cấp theo cấu trúc Claude 2026: tránh thảm họa, quy tắc nền tảng, đạo đức miền, hữu ích.

## Bài tập

1. Chạy `code/main.py`. So sánh tỷ lệ token có hại của model cơ sở với phiên bản được huấn luyện bởi CAI. Cần bao nhiêu bước sửa đổi để đạt được số không?

2. Đọc hiến pháp năm 2026 của Anthropic (anthropic.com/news/claudes-constitution). Liệt kê một nguyên tắc sẽ xếp hạng Cấp 1 và một nguyên tắc sẽ xếp hạng Cấp 4. Tại sao cấu trúc ưu tiên lại quan trọng đối với xung đột?

3. Thiết kế hiến pháp cho một trợ lý lập trình AI. Chỉ định Cấp 1 (thảm họa: lệnh hủy diệt mà không có sự chấp thuận), Cấp 2, Cấp 3, Cấp 4. Giữ mỗi bậc theo 3-5 nguyên tắc.

4. CAI thay thế máy dán nhãn của con người bằng máy dán nhãn AI. Đặt tên cho một chế độ lỗi giống như sự cố vẫn có thể xảy ra trong RLAIF và thiết kế phát hiện cho chế độ đó.

5. Đọc Phương pháp luận của Bộ phân loại Hiến pháp v2 (nếu có). Giải thích lý do tại sao chi phí điện toán ~1% là một câu chuyện an toàn khác về chất lượng so với 23,7%.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| AI hiến pháp | "AI được huấn luyện với nguyên tắc" | pipeline hai giai đoạn: tự phê bình và sửa đổi SFT, sau đó RL từ phản hồi AI |
| RLAIF | "RLHF không có con người" | RL với các tùy chọn được tạo bởi trình gắn nhãn AI; rest của pipeline không thay đổi |
| Hiến pháp | "Các nguyên tắc" | Một danh sách các quy tắc ngôn ngữ tự nhiên được sắp xếp theo thứ tự mà critique/labeler model tham khảo |
| Phê bình và sửa đổi | "vòng lặp SFT" | Đưa ra phản hồi → phê bình theo nguyên tắc → sửa đổi mục tiêu SFT → |
| Bộ phân loại hiến pháp | "cổng đầu ra" | Bộ phân loại nhẹ đánh giá đầu ra dựa trên hiến pháp và blocks/logs |
| Ưu tiên bốn cấp | "Người giải quyết xung đột" | Hệ thống phân cấp hiến pháp Claude năm 2026: Nền tảng > thảm khốc > đạo đức > hữu ích |
| Phản hồi model | "Máy dán nhãn AI" | model đọc một nguyên tắc và xếp hạng một cặp hoàn thành |

## Đọc thêm

- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback (arXiv:2212.08073)](https://arxiv.org/abs/2212.08073) - pipeline hai pha ban đầu
- [Anthropic — Claude's Constitution (Jan 2026)](https://www.anthropic.com/news/claudes-constitution) - viết lại bốn cấp năm 2026, CC0 1.0
- [Anthropic — Constitutional Classifiers (2024-2026)](https://www.anthropic.com/research/constitutional-classifiers) - phòng thủ cổng đầu ra với ~1% chi phí trong v2
- [Lee et al. — RLAIF vs RLHF: Scaling Reinforcement Learning from Human Feedback (arXiv:2309.00267)](https://arxiv.org/abs/2309.00267) — so sánh RLAIF / RLHF thực nghiệm
- [Kundu et al. — Specific versus General Principles for Constitutional AI (arXiv:2310.13798)](https://arxiv.org/abs/2310.13798) - ảnh hưởng của độ chi tiết nguyên tắc
