# Nghiên cứu Alignment tự động (Anthropic AAR)

> Anthropic điều hành các nhóm song song của các nhà nghiên cứu Alignment tự trị Claude Opus 4.6 trong các sandboxes độc lập, phối hợp thông qua một diễn đàn chung có nhật ký nằm bên ngoài bất kỳ sandbox nào (vì vậy agents không thể xóa hồ sơ của chính họ). Về vấn đề training yếu đến mạnh, AAR vượt trội hơn các nhà nghiên cứu con người. Cờ tóm tắt riêng của Anthropic cho thấy quy trình làm việc được quy định thường hạn chế tính linh hoạt của AAR và làm giảm hiệu suất. Tự động hóa nghiên cứu alignment là bước nén nén dòng thời gian đến các rủi ro sai lệch chính xác mà RSP muốn phát hiện.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, parallel-research-forum simulator)
**Kiến thức tiên quyết:** Giai đoạn 15 · 05 (AI Scientist v2), Giai đoạn 15 · 04 (Tổng giám đốc)
**Thời lượng:** ~60 phút

## Vấn đề

Nghiên cứu Alignment tốn kém trong thời gian của con người và nhà nghiên cứu. Các vấn đề như giám sát có thể mở rộng, đặc tả phần thưởng hoặc training yếu đến mạnh yêu cầu thử nghiệm mất hàng tuần cho mỗi lần lặp. Khi các khả năng tiên tiến tiến lên, khối lượng công việc alignment tăng nhanh hơn so với nguồn cung cấp các nhà nghiên cứu có trình độ.

Nghiên cứu Alignment tự động (AAR) đặt câu hỏi liệu cùng một biên giới models có khả năng vượt xa alignment có thể góp phần thu hẹp khoảng cách hay không. Báo cáo năm 2026 của Anthropic về nghiên cứu từ yếu đến mạnh training (alignment.anthropic.com/2026/automated-w2s-researcher/) do AAR thực hiện là một trong những kết quả công khai đầu tiên từ một hệ thống được triển khai của class này.

Kết quả thực sự tích cực: AAR giải quyết một vấn đề nghiên cứu tốt hơn so với đường cơ sở của con người. Kết quả cũng tập trung, trong một hệ thống duy nhất, mọi mối quan tâm mà giai đoạn này đã phát triển. Nếu nghiên cứu alignment có thể được tự động hóa, thì các bộ phận ảnh hưởng đến các biện pháp bảo vệ cũng vậy. Các ngưỡng của RSP cho AI năng lực R&D được viết với vòng lặp này.

## Khái niệm

### Kiến trúc (như được mô tả công khai)

- **Parallel agents.** Nhiều phiên bản Claude Opus 4.6 chạy dưới dạng AAR, mỗi phiên bản trong một sandbox độc lập. Sandboxes giới hạn hệ thống tệp, mạng và bề mặt process.
- **Diễn đàn được chia sẻ.** Các phát hiện được xuất bản trên một diễn đàn được chia sẻ có phần phụ trợ lưu trữ nằm bên ngoài mỗi sandbox. Agents có thể đọc diễn đàn; agents không thể xóa hoặc chỉnh sửa bản ghi diễn đàn từ trong sandbox của riêng họ.
- **Phân bổ nhiệm vụ.** Hàng đợi nhiệm vụ do con người thiết kế gán các vấn đề nghiên cứu cho AAR. AAR đề xuất phân hủy, chạy thí nghiệm, viết kết quả.
- **Tập trung training yếu đến mạnh.** Nghiên cứu được công bố đã đề cập đến training yếu đến mạnh - sử dụng một model giám sát yếu hơn để huấn luyện một model học sinh mạnh mẽ hơn. AAR vượt trội hơn các nhà nghiên cứu con người trong nhiệm vụ này.

### Tại sao nhật ký ngoài sandbox lại quan trọng

Thuộc tính toàn vẹn nhật ký là sự khác biệt giữa thử nghiệm có kiểm soát và thử nghiệm có khả năng bị xâm phạm. Nếu AAR có thể chỉnh sửa nhật ký của riêng họ, họ có thể che đậy các thí nghiệm thất bại hoặc đáng ngờ. Anthropic đánh dấu điều này trực tiếp: thuộc tính là một điều kiện cần thiết để tin tưởng vào kết quả nghiên cứu và việc xác minh không phải là điều tầm thường.

Tương tự về mặt cơ sở hạ tầng là nhật ký chỉ nối với ghi vào bộ nhớ ngoài (ví dụ: khóa đối tượng S3, nhật ký Datadog bất biến). agent không có thông tin xác thực để sửa đổi các bản ghi trước đây. Đây là mẫu tương tự mà các nhóm bảo mật của công ty năm 2026 sử dụng để tạo các dấu vết kiểm tra chống ransomware.

### Sự đánh đổi quy trình làm việc theo quy định

Báo cáo của Anthropic lưu ý rằng quy trình làm việc được quy định thường hạn chế tính linh hoạt của AAR và làm giảm hiệu suất. Nói cách khác: nếu nhà thiết kế con người viết kế hoạch nghiên cứu từng bước, AAR sẽ làm tồi tệ hơn so với khi được đưa ra các mục tiêu cấp cao và tự do phân hủy. Điều này phù hợp với kết quả trong agentic nghiên cứu (LangGraph, Agent Framework của Microsoft) - giàn giáo được chỉ định quá mức làm tổn thương agents có khả năng.

Ý nghĩa an toàn là tinh tế. Tính linh hoạt giúp cải thiện hiệu suất AAR cũng là tính linh hoạt cho phép phân hủy không được chấp thuận, sử dụng công cụ mới và trôi mục tiêu. Sự đánh đổi là một quyết định về không gian thiết kế: bao nhiêu linh hoạt mua đủ khả năng để đánh bại con người, mà không cần trao cho agent không gian để khái quát hóa các mục tiêu ngoài phân phối?

### Rủi ro nén

RSP v3.0 (Bài 19) giới thiệu ngưỡng năng lực R&D AI: khả năng tự động hóa hoàn toàn AI pipeline R&D với chi phí cạnh tranh so với con người + công cụ AI. FSF v3 của DeepMind bao gồm mức độ tự chủ ML R & D tương tự. Cả hai frameworks đều coi ngưỡng này là trigger cho các biện pháp kiểm soát nâng cao.

AAR còn thiếu một bước so với ngưỡng: nó tự động hóa một phần của pipeline (alignment nghiên cứu về các nhiệm vụ cụ thể, có phạm vi tốt) nhưng không phải vòng lặp phát triển khả năng từ đầu đến cuối. Câu hỏi về dòng thời gian là khoảng cách thu hẹp nhanh như thế nào.

Các mốc thời gian bị nén là mối quan tâm kép-thất bại. Nếu alignment nghiên cứu và nghiên cứu năng lực kết hợp với tốc độ tương tự, bề mặt rủi ro sai lệch sẽ tăng lên ít nhất là nhanh như khả năng. Nếu khả năng kết hợp nhanh hơn (xu hướng lịch sử), khoảng cách sẽ mở rộng. Đây là lập luận cho AAR là một hàng hóa đủ điều kiện: mỗi kết quả alignment bổ sung làm giảm khoảng cách nếu và chỉ khi process nghiên cứu đáng tin cậy.

### Những gì AAR không thay thế

Các nhà nghiên cứu con người thiết lập hàng đợi nhiệm vụ, xem xét kết quả và nắm giữ thẩm quyền hiến pháp. AAR tăng tốc ở giữa pipeline chứ không phải phần cuối. Kết quả đầu ra đã xuất bản của Anthropic bao gồm cả đóng góp của AAR và đánh giá của nhà nghiên cứu con người về những gì cần xuất bản, những gì cần rút lại và những gì cần tinh chỉnh.

Điều này phù hợp với mô hình đề xuất sau đó commit từ Bài 15 được áp dụng cho chính nghiên cứu: AAR đề xuất; con người commit.

## Ứng dụng

`code/main.py` mô phỏng một diễn đàn nghiên cứu đa AAR. Ba "agents nghiên cứu" độc lập chạy song song, đăng các phát hiện lên một diễn đàn chỉ bổ sung với các nỗ lực giả mạo tùy chọn.

Xem:

- Cách ghi nhật ký ngoài sandbox phát hiện các nỗ lực giả mạo nhật ký.
- Cách phân bổ nhiệm vụ với phân tách tự do vượt trội hơn quy trình làm việc cố định đối với các vấn đề khó hơn.
- Cách một AAR bị xâm phạm có thể làm ô nhiễm tín hiệu của diễn đàn nếu nhật ký không rõ ràng là giả mạo.

## Sản phẩm bàn giao

`outputs/skill-aar-deployment-review.md` là đánh giá trước khi triển khai pipeline nghiên cứu alignment tự động.

## Bài tập

1. Chạy `code/main.py`. So sánh cài đặt "quy trình làm việc cố định" và "phân hủy tự do". Phân hủy tự do có luôn chiến thắng hay có vấn đề classes khi quy trình làm việc cố định cạnh tranh?

2. Sửa đổi trình mô phỏng để một người agent cố gắng giả mạo nhật ký. Xác nhận nhật ký chỉ nối đã phát hiện ra nó. Viết một đoạn mô tả chính xác chữ ký phát hiện trông như thế nào trong nhật ký.

3. Đọc báo cáo AAR yếu đến mạnh của Anthropic. Xác định nhiệm vụ phụ cụ thể mà AAR đánh bại các nhà nghiên cứu con người. Điều gì khiến nó phù hợp với tự động hóa?

4. Thiết kế policy phân bổ hàng đợi tác vụ cân bằng tính linh hoạt của AAR (kết quả tốt hơn) với các ràng buộc quy trình làm việc theo quy định (kiểm tra dễ dàng hơn). Mô tả cách bạn sẽ A/B kiểm tra cả hai.

5. Đọc ngưỡng R&D-3.0 AI của RSP v4. Trong một đoạn, hãy mô tả những gì bạn nghĩ sẽ vượt qua nó mà AAR hiện không có.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| AAR | "Nhà nghiên cứu Alignment tự động" | Claude phiên bản Opus 4.6 hoạt động tự động trên các vấn đề alignment |
| training yếu đến mạnh | "Training một model mạnh hơn với một người giám sát yếu hơn" | AAR benchmark giám sát có thể mở rộng cổ điển vượt trội hơn con người trên |
| Diễn đàn chia sẻ | "Nơi agents công bố kết quả" | Chỉ bổ sung, bộ nhớ ngoài sandbox |
| Nhật ký ngoài sandbox | "Agent không thể chỉnh sửa bản ghi của chính mình" | Ghi qua rõ ràng giả mạo vào bộ nhớ ngoài |
| Quy trình làm việc theo quy định | "Kế hoạch từng bước từ nhà thiết kế con người" | Hạn chế AAR; thường làm giảm hiệu suất so với phân hủy tự do |
| Phân hủy tự do | "Agent quyết định làm thế nào để phá vỡ nhiệm vụ" | Có năng lực hơn, khó kiểm tra hơn |
| AI Ngưỡng R&D | "Mức năng lực RSP/FSF" | Tự động hóa hoàn toàn pipeline R&D với chi phí cạnh tranh |
| Dòng thời gian nén | "Cuộc đua Alignment và năng lực" | Nếu khả năng kết hợp nhanh hơn alignment, nguy cơ sai lệch sẽ tăng lên |

## Đọc thêm

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — nguồn chính.
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — AI khung ngưỡng R&D.
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) - khung agent tự chủ rộng hơn.
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — ML cấp độ tự chủ R&D song song với RSP.
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/) - vấn đề cơ bản mà AAR tấn công.
