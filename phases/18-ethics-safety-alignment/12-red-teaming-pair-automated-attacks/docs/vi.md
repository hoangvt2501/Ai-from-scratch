# Đội đỏ: PAIR và các cuộc tấn công tự động

> Chao, Robey, Dobriban, Hassani, Pappas, Wong (NeurIPS 2023, arXiv: 2310.08419). PAIR - Prompt Automatic Iterative Refinement - là bẻ khóa hộp đen tự động chính tắc. Một kẻ tấn công LLM với một đội đỏ system prompt lặp đi lặp lại đề xuất bẻ khóa cho một LLM mục tiêu, tích lũy các lần thử và phản hồi trong lịch sử trò chuyện của chính nó dưới dạng phản hồi theo ngữ cảnh. PAIR thường thành công trong vòng 20 truy vấn, hiệu quả hơn GCG (tìm kiếm gradient cấp token của Zou và cộng sự) và không yêu cầu quyền truy cập hộp trắng. PAIR hiện là đường cơ sở tiêu chuẩn trong JailbreakBench (arXiv: 2404.01318) và HarmBench, cùng với GCG, AutoDAN, TAP và Prompt đối thủ thuyết phục.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, mock PAIR loop against a toy target)
**Kiến thức tiên quyết:** Giai đoạn 18 · 01 (theo hướng dẫn), Giai đoạn 14 (kỹ thuật agent)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Mô tả thuật toán PAIR: system prompt kẻ tấn công, tinh chỉnh lặp lại, phản hồi theo ngữ cảnh.
- Giải thích lý do tại sao PAIR hiệu quả hơn GCG khi mục tiêu là hộp đen.
- Kể tên bốn đường cơ sở tấn công tự động khác (GCG, AutoDAN, TAP, PAP) và nêu một feature phân biệt của mỗi đường cơ sở.
- Mô tả các giao thức đánh giá JailbreakBench và HarmBench và "tỷ lệ tấn công thành công" có nghĩa là gì trong mỗi giao thức.

## Vấn đề

Đội đỏ từng là một hoạt động thủ công. Một số lượng nhỏ các chuyên gia thử nghiệm đã xây dựng prompts đối nghịch và theo dõi cái nào hoạt động. Điều này không mở rộng: tỷ lệ tấn công thành công cần một mẫu thống kê và mục tiêu là mục tiêu di chuyển với mỗi lần phát hành model. PAIR vận hành đội đỏ như một vấn đề tối ưu hóa với mục tiêu hộp đen.

## Khái niệm

### Thuật toán PAIR

Đầu vào:
- Mục tiêu LLM T (model chúng ta đang tấn công).
- Judge LLM J (chấm điểm xem một câu trả lời có phải là bẻ khóa hay không).
- Kẻ tấn công LLM A (đội optimizer đỏ).
- Chuỗi mục tiêu G: "phản ứng bằng [hướng dẫn có hại]."
- Ngân sách K (thường là 20 truy vấn).

Vòng lặp, cho k trong 1..K:
1. A được nhắc với mục tiêu G và lịch sử của các cặp (prompt, phản hồi) cho đến nay.
2. A phát ra một prompt p_k mới.
3. Gửi p_k cho T; nhận phản hồi r_k.
4. J ghi bàn (p_k, r_k) vào khung thành.
5. Nếu điểm > = ngưỡng, hãy tạm dừng - tìm thấy bẻ khóa.
6. Nếu không, thêm (p_k, r_k) vào lịch sử của A; tiếp tục.

Kết quả thực nghiệm (NeurIPS 2023): Tỷ lệ tấn công thành công >50% chống lại GPT-3.5-turbo, Llama-2-7B-chat; truy vấn trung bình để thành công trong phạm vi 10-20.

### Tại sao PAIR hiệu quả

GCG (Zou et al. 2023) tìm kiếm các hậu tố token đối nghịch theo gradient; Nó yêu cầu quyền truy cập model hộp trắng và tạo ra các hậu tố không thể đọc được. PAIR là hộp đen và tạo ra các cuộc tấn công ngôn ngữ tự nhiên truyền qua models. Phản hồi theo ngữ cảnh của PAIR cho phép kẻ tấn công học hỏi từ mỗi lần từ chối; GCG không có tính năng tương đương (mỗi bản cập nhật token mới phải khám phá lại tiến trình prior).

### Các cuộc tấn công tự động liên quan

- **GCG (Zou et al. 2023, arXiv:2307.15043).** gradient cấp Token tìm kiếm các hậu tố đối thủ. Hộp trắng, có thể chuyển nhượng, tạo ra các chuỗi không thể đọc được.
- **AutoDAN (Liu et al. 2023).** Tìm kiếm tiến hóa trên prompts, được hướng dẫn bởi một mục tiêu phân cấp.
- **TAP (Mehrotra et al. 2024).** Cây tấn công với cắt tỉa - branches nhiều rollouts kiểu PAIR.
- **PAP (Zeng et al. 2024).** Prompts đối thủ thuyết phục - mã hóa các kỹ thuật thuyết phục của con người dưới dạng prompt mẫu.

### JailbreakBench và HarmBench

Cả hai (2024) đều chuẩn hóa đánh giá:

- JailbreakBench (arXiv: 2404.01318). 100 hành vi có hại trên 10 danh mục OpenAI policy. Tỷ lệ tấn công thành công (ASR) là thước đo chính. Yêu cầu một thẩm phán (GPT-4-turbo, Llama Guard hoặc StrongREJECT).
- HarmBench (Mazeika và cộng sự 2024). 510 hành vi trên 7 danh mục, với các bài kiểm tra tác hại về ngữ nghĩa và chức năng. So sánh 18 đòn tấn công với 33 models.

ASR thường được báo cáo ở ngân sách truy vấn cố định. So sánh các cuộc tấn công đòi hỏi ngân sách phù hợp; ASR 90% ở 200 truy vấn không thể so sánh với 85% ASR ở 20.

### Lý do quan trọng đối với việc triển khai năm 2026

Mọi phòng thí nghiệm biên giới hiện chạy PAIR và TAP với production models trước khi phát hành. ASR quỹ đạo xuất hiện trong thẻ model (Bài 26) và phụ lục trường hợp an toàn (Bài 18). Cuộc tấn công không phải là kỳ lạ - nó là cơ sở hạ tầng tiêu chuẩn.

### Điều này phù hợp với Giai đoạn 18 ở đâu

Bài 12 là nền tảng tấn công tự động. Bài 13 (Jailbreak nhiều lần) là một khai thác độ dài bổ sung. Bài 14 (ASCII Art / Visual) là một cuộc tấn công mã hóa. Bài 15 (Indirect Prompt Injection) là bề mặt tấn công production năm 2026. Bài 16 bao gồm các đối tác công cụ phòng thủ (Llama Guard, Garak, PyRIT).

## Ứng dụng

`code/main.py` xây dựng một vòng lặp PAIR đồ chơi. Mục tiêu là một bộ phân loại giả từ chối prompts có hại "rõ ràng" (bộ lọc từ khóa). Kẻ tấn công là một công cụ tinh chỉnh dựa trên quy tắc cố gắng diễn giải, đóng khung nhập vai và mã hóa. Giám khảo chấm điểm câu trả lời. Bạn xem kẻ tấn công thành công trong ~5-15 lần lặp lại đối với bộ lọc từ khóa và thất bại trước bộ lọc ngữ nghĩa.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-attack-audit.md`. Với một báo cáo đánh giá nhóm đỏ, nó kiểm tra: cuộc tấn công nào đã được thực hiện (PAIR, GCG, TAP, AutoDAN, PAP), với ngân sách bao nhiêu, với thẩm phán nào, tập hợp hành vi có hại nào (JailbreakBench, HarmBench, nội bộ).

## Bài tập

1. Chạy `code/main.py`. Đo lường truy vấn trung bình để thành công cho ba chiến lược tấn công tích hợp. Giải thích giả định phòng thủ mục tiêu nào mà mỗi người khai thác.

2. Thực hiện chiến lược tấn công thứ tư (ví dụ: dịch sang ngôn ngữ khác, mã hóa base64). Báo cáo truy vấn trung bình mới để thành công dựa trên mục tiêu bộ lọc từ khóa và mục tiêu bộ lọc ngữ nghĩa.

3. Đọc Chao et al. 2023 Hình 5 (so sánh PAIR và GCG). Mô tả hai tình huống mà GCG được ưa chuộng mặc dù lợi thế hiệu quả của PAIR.

4. JailbreakBench báo cáo ASR chống lại một bộ mục tiêu cố định. Thiết kế một số liệu bổ sung để đo lường sự đa dạng của cuộc tấn công (variance trong prompts thành công). Giải thích lý do tại sao sự đa dạng lại quan trọng đối với đánh giá quốc phòng.

5. TAP (Mehrotra 2024) mở rộng PAIR với phân nhánh + cắt tỉa. Phác thảo tiện ích mở rộng kiểu TAP để `code/main.py` và mô tả chi phí tính toán so với tỷ lệ thành công.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| CẶP | "bẻ khóa tự động" | Prompt tinh chỉnh lặp đi lặp lại tự động; Vòng lặp LLM kẻ tấn công + LLM thẩm phán |
| GCG | "gradient bẻ khóa" | Hộp trắng token cấp gradient tìm kiếm hậu tố đối nghịch |
| Tỷ lệ tấn công thành công (ASR) | "% bẻ khóa tại K truy vấn" | Số liệu chính; phải được báo cáo với ngân sách truy vấn và danh tính thẩm phán |
| Thẩm phán LLM | "Người ghi bàn" | LLM điều đó đánh giá xem một phản ứng có thỏa mãn mục tiêu có hại hay không |
| Băng ghế bẻ khóa | "Đánh giá" | Tập hợp hành vi gây hại được chuẩn hóa với các danh mục được gắn thẻ |
| Băng ghế HarmBench | "Băng ghế rộng hơn" | 510 hành vi, kiểm tra tác hại chức năng + ngữ nghĩa |
| CHẠM VÀO | "Cây tấn công" | PAIR với phân nhánh + cắt tỉa; ASR tốt hơn ở điện toán cao hơn |

## Đọc thêm

- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — Giấy PAIR, NeurIPS 2023
- [Zou et al. — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — Giấy GCG
- [Chao et al. — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) - đánh giá tiêu chuẩn hóa
- [Mazeika et al. — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) - đánh giá rộng hơn
