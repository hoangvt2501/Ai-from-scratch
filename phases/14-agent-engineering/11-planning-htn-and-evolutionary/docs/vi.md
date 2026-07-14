# Lập kế hoạch với HTN và Evolutionary Search

> Lập kế hoạch tượng trưng xử lý các trường hợp mà kế hoạch là chính xác. Tìm kiếm mã tiến hóa xử lý các trường hợp chức năng thể dục có thể kiểm tra bằng máy. ChatHTN (2025) và AlphaEvolve (2025) hiển thị những gì mỗi loại mở khóa khi được ghép nối với LLM.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib)
**Kiến thức tiên quyết:** Giai đoạn 14 · 02 (ReWOO và Lập kế hoạch và Thực hiện)
**Thời lượng:** ~75 phút

## Mục tiêu học tập

- Giải thích Mạng tác vụ phân cấp: nhiệm vụ, phương pháp, toán tử, điều kiện tiên quyết, hiệu ứng.
- Mô tả vòng lặp lai của ChatHTN - tìm kiếm tượng trưng với LLM phân hủy dự phòng.
- Giải thích vòng lặp tiến hóa của AlphaEvolve và lý do tại sao nó chỉ hoạt động với một công cụ đánh giá lập trình.
- Triển khai một công cụ lập kế hoạch HTN đồ chơi cộng với tìm kiếm tiến hóa đồ chơi trong stdlib.

## Vấn đề

ReWOO (Bài 02), Lập kế hoạch và Thực hiện và Hành động lại bao gồm hầu hết các agent lập kế hoạch. Hai trường hợp họ không bao quát tốt:

1. **Kế hoạch có tính đúng đắn có thể chứng minh được.** Lập kế hoạch, đường bay, quy trình tuân thủ - kế hoạch phải hợp lý theo cấu trúc. Một kế hoạch LLM trôi chảy đôi khi ảo giác một bước là không thể chấp nhận được.
2. **Tối ưu hóa với chức năng thể dục có thể kiểm tra bằng máy.** Phép nhân ma trận, phỏng đoán lập lịch, trình biên dịch vượt qua - mục tiêu không phải là "một kế hoạch chính xác" mà là "kế hoạch tốt nhất".

Lập kế hoạch HTN và AlphaEvolve giải quyết hai vấn đề khác nhau. Cả hai đều sử dụng LLMs làm bộ khuếch đại, không thay thế.

## Khái niệm

### Mạng nhiệm vụ phân cấp

HTN là:

- **Tasks** — compound (sẽ được phân tách) và primitive (có thể thực thi trực tiếp).
- **Phương pháp** — cách phân tách một nhiệm vụ phức hợp thành các nhiệm vụ con, với các điều kiện tiên quyết.
- **Toán tử** — primitive hành động với các điều kiện tiên quyết và hiệu ứng.
- **Trạng thái** — một tập hợp các sự kiện.

Lập kế hoạch: cho một nhiệm vụ mục tiêu và trạng thái ban đầu, tìm một sự phân tách thành các toán tử primitive có các điều kiện tiên quyết được thỏa mãn theo trình tự.

HTN cũ hơn LLMs và vẫn là tài liệu tham khảo cho các kế hoạch chính xác có thể chứng minh.

### ChatHTN (Gopalakrishnan và cộng sự, 2025)

ChatHTN (arXiv:2505.11814) xen kẽ HTN tượng trưng với LLM truy vấn:

1. Cố gắng phân tách nhiệm vụ phức hợp hiện tại bằng các phương thức hiện có.
2. Nếu không có phương pháp nào áp dụng, hãy hỏi LLM: "Làm thế nào bạn sẽ phân hủy `task` trong trạng thái `s`?"
3. Dịch câu trả lời LLM thành các nhiệm vụ phụ của ứng viên.
4. Xác thực dựa trên schema của nhà điều hành; từ chối các phân hủy không hợp lệ.
5. Đệ quy.

Tuyên bố trung tâm của bài báo: mọi kế hoạch được đưa ra đều hợp lý vì các đề xuất LLM chỉ được đưa vào dưới dạng phân hủy ứng cử viên, không bao giờ là chỉnh sửa kế hoạch trực tiếp. Lớp biểu tượng sở hữu tính đúng đắn; LLM mở rộng thư viện phương thức.

Học phương pháp trực tuyến (OpenReview `gwYEDY9j2x`, 2025 theo dõi) thêm một người học khái quát hóa các phân tách do LLM tạo ra bằng hồi quy - cắt giảm tần suất truy vấn LLM lên đến 75%.

### AlphaEvolve (Novikov và cộng sự, 2025)

AlphaEvolve (arXiv: 2506.13131, DeepMind, tháng 6 năm 2025) là một con quái vật khác: tìm kiếm mã tiến hóa được dàn dựng bởi một nhóm Flash/Pro Gemini 2.0.

Vòng lặp:

1. Bắt đầu với một chương trình hạt giống + một công cụ đánh giá có lập trình (trả về điểm thể dục).
2. Quần thể LLMs đề xuất đột biến.
3. Chạy đột biến thông qua trình đánh giá.
4. Giữ những gì tốt nhất; đột biến một lần nữa.

Chiến thắng được công bố:

- Cải tiến đầu tiên so với Strassen cho phép nhân ma trận phức 4x4 trong 56 năm (48 phép nhân vô hướng).
- 0,7% đã khôi phục điện toán của Google thông qua phỏng đoán lập lịch Borg.
- Tăng tốc 32% FlashAttention trên khối lượng công việc biên giới.

Hạn chế cứng: chức năng thể dục phải có thể kiểm tra bằng máy. Tìm kiếm tiến hóa trên các câu trả lời văn xuôi không hội tụ.

### Khi nào sử dụng

| Vấn đề class | Sử dụng | Tại sao |
|---------------|-----|-----|
| Lập lịch với các ràng buộc cứng | HTN + Trò chuyệnHTN | Độ bền có thể chứng minh được |
| Tối ưu hóa trình biên dịch | AlphaEvolve | Thể lực có thể kiểm tra bằng máy |
| Thực hiện nhiệm vụ nhiều bước | ReAct / ReWOO | LLM trong vòng lặp, không có đảm bảo chính thức |
| Cải thiện mã với các bài kiểm tra | AlphaEvolve | Các bài kiểm tra là người đánh giá |
| Tự động hóa ràng buộc Policy | HTN | Điều kiện tiên quyết mã hóa policy |

### Mô hình này sai ở đâu

- **HTN không có người vận hành.** Nếu không có precondition/effect schemas, tuyên bố về tính hợp lý sẽ sụp đổ. "LLM gợi ý phân rã" của ChatHTN yêu cầu schema từ chối các nước đi không hợp lệ.
- **AlphaEvolve mà không có người đánh giá thực sự.** "Hỏi LLM xem mã có tốt hơn không" không phải là một chức năng thể dục. Người đánh giá phải xác định và nhanh chóng.
- **Kỹ thuật quá mức.** Hầu hết các nhiệm vụ agent cũng không cần. Tiếp cận ReAct hoặc ReWOO trước.

## Tự xây dựng

`code/main.py` thực hiện hai đồ chơi:

- Một công cụ lập kế hoạch HTN stdlib với các toán tử, phương thức, điều kiện tiên quyết, hiệu ứng và một `LLMFallback` bắt đầu khi không có phương pháp nào phù hợp với một nhiệm vụ phức hợp. "LLM" là một trình phân hủy theo kịch bản nên trình lập kế hoạch chạy ngoại tuyến.
- Tìm kiếm tiến hóa stdlib qua các chương trình số học: phát triển các biểu thức có đầu ra giảm thiểu `|f(x) - target|` trong một tập thử nghiệm. Người đánh giá là xác định.

Chạy nó:

```
python3 code/main.py
```

Bản trace cho thấy công cụ lập kế hoạch HTN phân tách một nhiệm vụ phức hợp (với một kế hoạch giữa LLM dự phòng) và vòng lặp tiến hóa hội tụ trên một biểu thức mục tiêu.

## Ứng dụng

- **Công cụ lập kế hoạch HTN** — `pyhop`, `SHOP3` hoặc xây dựng của riêng bạn để thực thi policy theo tên miền cụ thể.
- **ChatHTN** — mã nghiên cứu; mô hình (biểu tượng + dự phòng LLM) chuyển rõ ràng cho bất kỳ công cụ lập kế hoạch HTN nào.
- **AlphaEvolve** — Bài báo DeepMind; Mẫu (Ensemble + Evaluator) có thể tái tạo. OpenEvolve và các fork mã nguồn mở tương tự đang nổi lên.
- **Agent frameworks** — chưa ship HTN hoặc AlphaEvolve class đầu tiên. Xây dựng nó như một subagent hoặc một worker nền.

## Sản phẩm bàn giao

`outputs/skill-hybrid-planner.md` tạo ra một giàn giáo lập kế hoạch lai (HTN hoặc tiến hóa) với vai trò LLM được xác định rõ ràng.

## Bài tập

1. Mở rộng công cụ lập kế hoạch HTN bằng cách quay ngược: khi điều kiện sau của người vận hành không thành công ở runtime, hãy quay lại và thử phương pháp tiếp theo.
2. Thêm bộ nhớ đệm phương thức LLM vào ChatHTN: khi LLM phân tách tác vụ `T` trong `P` mẫu trạng thái, hãy lưu trữ kết quả. Kiểm tra lại thư viện phương thức trước trong cuộc gọi tiếp theo.
3. Hoán đổi công cụ đánh giá tìm kiếm tiến hóa thành một bộ thử nghiệm thực. Phát triển một hàm sắp xếp vượt qua 20 trường hợp thử nghiệm; báo cáo các thế hệ để hội tụ.
4. Đọc ghi chú thiết kế trình đánh giá của AlphaEvolve. Thiết kế một trình đánh giá cho một miền bạn quan tâm (SQL tối ưu hóa truy vấn, giảm thiểu bộ kiểm tra, triển khai YAML).
5. Kết hợp: sử dụng HTN để phân tách một nhiệm vụ phức hợp thành các nhiệm vụ con, sau đó sử dụng tìm kiếm tiến hóa trên toán tử primitive của mỗi nhiệm vụ con. Nó tỏa sáng ở đâu, nó thiết kế quá mức ở đâu?

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|----------------|------------------------|
| HTN | "Công cụ lập kế hoạch phân cấp" | Phân tích nhiệm vụ với toán tử, điều kiện tiên quyết, hiệu ứng |
| Phương pháp | "Quy tắc phân hủy" | Cách chia nhiệm vụ phức hợp thành các nhiệm vụ con |
| Nhà điều hành | "Hành động Primitive" | Bước bê tông với điều kiện tiên quyết và hiệu quả |
| Trò chuyệnHTN | "LLM + HTN" | Công cụ lập kế hoạch biểu tượng hỏi LLM khi không có phương thức nào phù hợp |
| AlphaEvolve | "Tìm kiếm mã tiến hóa" | Ensemble LLMs mã đột biến; Trình đánh giá xác định chọn |
| Chức năng thể dục | "Người đánh giá" | Điểm xác định, có thể kiểm tra bằng máy trên đầu ra |
| Học phương pháp trực tuyến | "Phân hủy LLM được lưu trong bộ nhớ đệm" | Cửa hàng + khái quát hóa kế hoạch LLM để cắt giảm chi phí truy vấn |

## Đọc thêm

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) — Công cụ lập kế hoạch kết hợp biểu tượng + LLM
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — tìm kiếm mã tiến hóa với đột biến LLM
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - khi nào nên tiếp cận với một công cụ lập kế hoạch so với một vòng lặp đơn giản
