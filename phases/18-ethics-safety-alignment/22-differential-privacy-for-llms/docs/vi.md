# Quyền riêng tư khác biệt cho LLMs

> DP-SGD vẫn là tiêu chuẩn - các bản cập nhật gradient được tiêm nhiễu cung cấp đảm bảo chính thức (epsilon, delta). Chi phí điện toán, bộ nhớ và tiện ích là đáng kể; fine-tuning DP hiệu quả parameter (LoRA + DP-SGD) là configuration phổ biến năm 2025 (ACM 2025). Hai bằng chứng căng thẳng: inference thành viên dựa trên canary (Duan và cộng sự, 2024) báo cáo thành công hạn chế đối với models ngôn ngữ; Trích xuất dữ liệu training (Carlini và cộng sự, 2021; Nasr và cộng sự, 2025) phục hồi khả năng ghi nhớ nguyên văn đáng kể. Độ phân giải (arXiv:2503.06808, tháng 3 năm 2025): khoảng cách nằm ở những gì được đo lường - canary được chèn so với dữ liệu "có thể trích xuất nhiều nhất". Các thiết kế canary mới cho phép MIA dựa trên loss mà không có models bóng và mang lại cuộc kiểm tra DP không tầm thường đầu tiên của một LLM được huấn luyện trên dữ liệu thực với đảm bảo DP thực tế. Các lựa chọn thay thế: PMixED (arXiv:2403.15638) — dự đoán riêng tại thời điểm inference thông qua sự kết hợp của các chuyên gia về các phân phối token tiếp theo; Tạo dữ liệu tổng hợp DP (Nghiên cứu của Google 2024). Tấn công mới nổi: Đảo ngược quyền riêng tư khác biệt thông qua Phản hồi LLM - rò rỉ điểm tin cậy.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, DP-SGD noise-injection and ε-δ accountant demonstration)
**Kiến thức tiên quyết:** Giai đoạn 01 · 09 (lý thuyết thông tin), Giai đoạn 10 · 01 (model training lớn)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Xác định quyền riêng tư khác biệt (epsilon, delta) và nêu công thức DP-SGD.
- Giải thích căng thẳng 2024-2025: canary MIA so với trích xuất dữ liệu training cho những bức tranh khác nhau.
- Mô tả PMixED và lý do tại sao dự đoán riêng tư theo thời gian inference là một giải pháp thay thế cho DP training.
- Mô tả sự đảo ngược quyền riêng tư khác biệt thông qua cuộc tấn công Phản hồi LLM.

## Vấn đề

LLMs ghi nhớ. Carlini et al. 2021 đã cho thấy ngôn ngữ production models tái tạo nguyên văn training văn bản theo yêu cầu. DP là biện pháp phòng thủ chính thức: huấn luyện sao cho đầu ra có thể chứng minh là không nhạy cảm với bất kỳ ví dụ training nào. Bằng chứng 2024-2025 cho thấy DP-SGD là cần thiết nhưng các giá trị ε được triển khai có thể không khớp với mối đe dọa model.

## Khái niệm

### (ε, δ) - quyền riêng tư khác biệt

Một thuật toán ngẫu nhiên M là (ε, δ) -DP nếu đối với bất kỳ hai datasets nào khác nhau trong một ví dụ và bất kỳ sự kiện nào S:
P(M(D) tính bằng S) <= e^ε * P(M(D') tính bằng S) + δ.

Giải thích: phân phối đầu ra đủ gần (được tham số hóa bằng ε) để không thể suy ra sự đóng góp của bất kỳ cá nhân đơn lẻ nào một cách đáng tin cậy, ngoại trừ với xác suất δ.

### DP-SGD

Abadi và cộng sự, 2016. Công thức tiêu chuẩn:
1. Lấy mẫu một batch nhỏ.
2. Tính toán gradients cho mỗi ví dụ.
3. Cắt từng gradient cho mỗi ví dụ đến ngưỡng C.
4. Tổng gradients đã cắt và thêm nhiễu Gaussian với std σ * C.
5. Sử dụng tổng nhiễu để cập nhật parameters.

Chi phí bảo mật được theo dõi bởi kế toán (Kế toán Moments, kế toán Rényi DP). Các giá trị ε được báo cáo trong tài liệu LLM rất khác nhau tùy theo model mối đe dọa, độ nhạy cảm của dữ liệu và mục tiêu tiện ích; Không có ε mặc định "an toàn" phổ biến. Các ví dụ được xuất bản span khoảng ε ≈ 1-10 trong một số cài đặt LLM training, nhưng đây chỉ là minh họa — không phải là mặc định được đề xuất. ε thấp hơn thường yêu cầu nhiều nhiễu hơn và có thể tăng loss tiện ích.

### LoRA + DP-SGD

DP-SGD đầy đủ của một model biên giới là điều cấm đoán. LoRA (Hu et al. 2022) giới hạn các bản cập nhật gradient đối với một bộ điều hợp nhỏ, giảm dung lượng lưu trữ gradient cho mỗi ví dụ. LoRA + DP-SGD là configuration phổ biến năm 2025. Đảm bảo DP áp dụng cho bộ chuyển đổi; model cơ sở được giữ cố định.

### Căng thẳng 2024-2025

Hai dòng bằng chứng:

- **Canary MIA (Duan et al. 2024).** Chèn các canary duy nhất vào dữ liệu training, đo lường xem kẻ tấn công inference thành viên có thể xác định chúng hay không. Báo cáo thành công hạn chế về models ngôn ngữ. Gợi ý MIA rất khó.
- **Trích xuất dữ liệu Training (Carlini 2021, Nasr et al. 2025).** Prompt model có tiền tố; đo lường xem nó có khôi phục nguyên văn văn bản từ training hay không. Báo cáo ghi nhớ đáng kể. Gợi ý MIA là dễ dàng theo nghĩa liên quan.

Nghị quyết tháng 3 năm 2025 (arXiv: 2503.06808): cả hai đo lường những thứ khác nhau. MIA hỏi "có phải ví dụ e trong D không?" trên canary được chèn vào. Extraction hỏi "tôi có thể phục hồi những gì của D?" Ví dụ "có thể trích xuất nhất" là những gì quan trọng đối với quyền riêng tư; Canary báo cáo điều này dưới mức vì chúng không được tối ưu hóa để có thể trích xuất được.

Thiết kế canary mới. MIA dựa trên Loss không có models bóng. Kiểm tra DP không tầm thường đầu tiên của một LLM trên dữ liệu thực với đảm bảo DP thực tế.

### Các lựa chọn thay thế cho DP training

- **PMixED (arXiv:2403.15638).** Dự đoán riêng tư tại thời điểm inference. Sự kết hợp của các chuyên gia về các bản phân phối token tiếp theo; mỗi chuyên gia nhìn thấy một mảnh dữ liệu training; tổng hợp thêm nhiễu cho DP. Tránh DP hoàn toàn training.
- **Tạo dữ liệu tổng hợp DP (Nghiên cứu của Google 2024).** LoRA-fine-tune với DP-SGD, lấy mẫu dữ liệu tổng hợp, huấn luyện bộ phân loại xuôi dòng trên dữ liệu tổng hợp.

Cả hai đều tránh chi phí tiện ích của training DP đầy đủ với cái giá phải trả là một model mối đe dọa khác.

### Đảo ngược quyền riêng tư khác biệt thông qua phản hồi LLM

Cuộc tấn công năm 2025 mới nổi. Sử dụng điểm tin cậy của model được huấn luyện DP như một nhà tiên tri để xác định lại các cá nhân. Ngay cả khi đầu ra không bị rò rỉ, phân phối tin cậy vẫn có thể.

Phòng thủ: không để lộ bí mật, hoặc truncate/quantize chúng trước khi tiếp xúc. Đây là một yêu cầu bổ sung ngoài (ε, δ)-DP training.

### Điều này phù hợp với Giai đoạn 18 ở đâu

Bài 20-21 là bias/fairness. Bài 22 là quyền riêng tư. Bài 23 là nguồn gốc thông qua watermarking. Bài 27 bao gồm lớp nguồn gốc dữ liệu quy định.

## Ứng dụng

`code/main.py` mô phỏng DP-SGD trên một dataset phân loại nhị phân đồ chơi. Bạn có thể quét hệ số nhiễu σ và định mức cắt C và theo dõi ngân sách (ε, δ) và chi phí accuracy. Một "cuộc tấn công canary" chèn một training duy nhấtample và đo lường liệu một bài kiểm tra log-loss có thể phát hiện nó trước và sau DP hay không.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-dp-audit.md`. Đưa ra yêu cầu DP về ngôn ngữ model triển khai, nó kiểm tra: các giá trị (ε, δ), kế toán được sử dụng, giao thức đánh giá MIA và liệu các vectors tiếp xúc với sự tin cậy đã được đánh giá hay chưa.

## Bài tập

1. Chạy `code/main.py`. Quét các σ trong {0.5, 1.0, 2.0} và báo cáo sự đánh đổi (ε, δ)-accuracy. Xác định điểm mà tiện ích sụp đổ.

2. Triển khai chèn canary và kiểm tra loss nhật ký. Đo tỷ lệ phát hiện trước và sau DP-SGD ở σ = 1.0.

3. Đọc Nasr et al. 2025 về trích xuất dữ liệu training. Tại sao thành công trong quá trình chiết xuất không sụp đổ dưới ε vừa phải? Điều này ngụ ý gì về MIA dưới dạng đánh giá?

4. Thiết kế triển khai bằng PMixED (arXiv:2403.15638) hoạt động hoàn toàn tại inference thời điểm. Mối đe dọa model mà PMixED giải quyết mà DP-SGD không giải quyết là gì?

5. Phác thảo DP Reversal thông qua đòn tấn công LLM Phản hồi. Thiết kế một biện pháp đối phó hạn chế rò rỉ điểm tin cậy và ước tính chi phí triển khai của nó.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| DP | "(ε, δ)-quyền riêng tư khác biệt" | Quyền riêng tư chính thức: phân phối đầu ra đóng theo sự thay đổi dataset lân cận |
| DP-SGD | "SGD phun nhiễu" | Gradient cắt + thêm nhiễu Gaussian; training DP tiêu chuẩn |
| LoRA + DP-SGD | "fine-tune tư nhân hiệu quả" | DP-SGD trên bộ điều hợp cấp thấp; Tiêu chuẩn 2025 configuration |
| MIA | "inference thành viên" | Tấn công xác định xem một ví dụ có trong dữ liệu training hay không |
| Canary | "ví dụ về hình mờ được chèn" | Ví dụ training duy nhất được sử dụng để đo rò rỉ DP |
| PMixED | "hỗn hợp inference riêng" | DP Inference lần thông qua sự kết hợp của các chuyên gia về các đợt phân phối token tiếp theo |
| Đảo ngược DP | "Tấn công rò rỉ niềm tin" | Tấn công sử dụng độ tin cậy của model làm oracle để xác định lại |

## Đọc thêm

- [Abadi et al. — DP-SGD (arXiv:1607.00133)](https://arxiv.org/abs/1607.00133) — thuật toán training DP tiêu chuẩn
- [Carlini et al. — Extracting Training Data (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805) — bài trích xuất kinh điển
- [Duan et al. — Canary MIA on LLMs (arXiv:2402.07841, 2024)](https://arxiv.org/abs/2402.07841) - MIA thành công hạn chế
- [Kowalczyk et al. — Auditing DP for LLMs (arXiv:2503.06808, March 2025)](https://arxiv.org/abs/2503.06808) - giải quyết căng thẳng
- [PMixED (arXiv:2403.15638)](https://arxiv.org/abs/2403.15638) — Dự đoán riêng tư inference lần
