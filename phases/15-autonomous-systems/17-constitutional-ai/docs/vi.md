# Ghi đè AI và quy tắc hiến pháp

> Hiến pháp Claude ngày 22 tháng 1 năm 2026 của Anthropic dài 79 trang và là CC0. Nó chuyển từ alignment dựa trên quy tắc sang dựa trên lý trí và thiết lập hệ thống phân cấp ưu tiên bốn cấp: (1) an toàn và hỗ trợ giám sát của con người, (2) đạo đức, (3) hướng dẫn Anthropic, (4) hữu ích. Các hành vi được chia thành các lệnh cấm được mã hóa cứng (nâng cao vũ khí sinh học, CSAM) mà người vận hành và người dùng không thể ghi đè và các mặc định được mã hóa mềm mà người vận hành có thể điều chỉnh trong giới hạn xác định. Bản gốc năm 2022 (Bai và cộng sự) đã rèn luyện tính vô hại thông qua tự phê bình và RLAIF chống lại hiến pháp. Cảnh báo trung thực: alignment dựa trên lý do dựa trên các nguyên tắc khái quát hóa model cho các tình huống không lường trước được. Thử nghiệm có sự tham gia năm 2023 của riêng Anthropic cho thấy sự khác biệt ~50% giữa các nguyên tắc có nguồn gốc công và doanh nghiệp; Phiên bản năm 2026 không kết hợp những phát hiện đó.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, four-tier priority resolver)
**Kiến thức tiên quyết:** Giai đoạn 15 · 06 (Nghiên cứu alignment tự động), Giai đoạn 15 · 10 (Chế độ quyền)
**Thời lượng:** ~60 phút

## Vấn đề

Một agent thực địa nhìn thấy đầu vào mà các nhà thiết kế của nó chưa bao giờ thấy. Không có danh sách quy tắc nào đủ dài để bao quát chúng. Không có danh sách quy tắc nào đủ ngắn để áp dụng nhanh chóng dưới áp lực tính toán. Câu hỏi thực tế: làm thế nào để bạn sắp xếp một agent với các nguyên tắc tồn tại cả một đuôi dài của các trường hợp và inference nhanh?

alignment dựa trên quy tắc (RBA): liệt kê mọi thứ không được phép. Kiểm tra nhanh, dễ kiểm tra, không thể cập nhật, thường từ chối quá mức đối với các thiết bị tương tự gần mà nó không lường trước được. alignment dựa trên lý do (Hiến pháp Claude 2026): mã hóa các nguyên tắc, để model lý luận. Quy mô trên các trường hợp không nhìn thấy, khó kiểm tra hơn, chế độ thất bại là áp dụng sai nguyên tắc hơn là bỏ lỡ quy tắc.

Hiến pháp năm 2026 có một vị trí trung gian rõ ràng. Các lệnh cấm được mã hóa cứng - những thứ có sai không phụ thuộc vào ngữ cảnh (nâng cao vũ khí sinh học, CSAM) - là RBA: không bao giờ, bất kể hướng dẫn của người vận hành hoặc người dùng. Mọi thứ khác đều dựa trên lý trí trong hệ thống phân cấp bốn tầng: an toàn và hỗ trợ giám sát của con người trước tiên; đạo đức thứ hai; Anthropic tuyên bố hướng dẫn thứ ba; hữu ích cuối cùng. Người vận hành có thể điều chỉnh mặc định trong vùng mã hóa mềm nhưng không thể chạm vào các lệnh cấm được mã hóa cứng.

## Khái niệm

### Hệ thống phân cấp ưu tiên bốn cấp

1. **An toàn và hỗ trợ giám sát của con người.** Cao nhất. model ưu tiên không làm suy yếu khả năng giám sát và sửa chữa AI của con người và Anthropic. Đây không phải là "thận trọng"; nó đặc biệt là "không hành động theo những cách khiến sự giám sát của con người trở nên khó khăn hơn".
2. **Đạo đức.** Trung thực, tránh làm hại con người, không lừa dối, không thao túng. Thay thế các hướng dẫn của Anthropic khi chúng xung đột.
3. **Anthropic hướng dẫn.** Các tiêu chuẩn hoạt động Anthropic quyết định vấn đề: phạm vi sản phẩm, mô hình tương tác, công cụ nào sẽ sử dụng khi nào.
4. **Hữu ích.** Thấp nhất. Hãy hữu ích nhất có thể trong các ưu tiên cao hơn.

Khi các cấp xung đột, cao hơn sẽ thắng. Đây là hình dạng tương tự như các ưu tiên của Unix hoặc QoS mạng - khung có nghĩa là tạo ra độ phân giải có thể dự đoán được, không nhất thiết phải là hành vi tốt nhất trên bất kỳ trục đơn lẻ nào.

### Các lệnh cấm được mã hóa cứng so với mặc định được mã hóa mềm

**Mã hóa cứng:**
- Vũ khí sinh học / nâng cao CBRN
- CSAM
- Các cuộc tấn công vào cơ sở hạ tầng quan trọng
- Lừa dối người dùng về danh tính của model khi được hỏi trực tiếp

Nhà điều hành không thể ghi đè những điều này. Người dùng không thể ghi đè những điều này. Chúng được thực thi ở cấp độ model trọng lượng nếu có thể (RLHF / AI training Hiến pháp) và ở cấp inference nếu không.

**Mặc định được mã hóa mềm (có thể điều chỉnh bởi người vận hành):**
- Độ dài phản hồi mặc định
- Phạm vi chủ đề (model có thể từ chối các chủ đề bên ngoài việc triển khai của nhà điều hành)
- Phong cách (trang trọng và giản dị)
- Các mẫu sử dụng công cụ

Các điều chỉnh toán tử xảy ra bên trong một giới hạn đã khai báo. Nhà điều hành không thể loại bỏ các lệnh cấm được mã hóa cứng bằng cách đổi tên chúng.

### The 2022 CAI training

AI Hiến pháp ban đầu (Bai et al., 2022) đã huấn luyện tính vô hại:

1. Tạo phản hồi cho một tập hợp các prompts.
2. Yêu cầu model phê bình từng phản ứng chống lại hiến pháp (các nguyên tắc rõ ràng).
3. Sửa đổi câu trả lời dựa trên lời phê bình.
4. RLAIF (học tăng cường từ phản hồi AI) trên các cặp đã sửa đổi.

Kết quả: một model từ chối các yêu cầu có hại với những lời giải thích có nguyên tắc, không phải từ chối toàn diện. Hiến pháp năm 2026 sử dụng hậu duệ của training này cộng với hậu training bổ sung trên hệ thống phân cấp cấp rõ ràng.

### Những alignment dựa trên lý do nào bắt và bỏ lỡ

**Bắt giữ:**
- Sự kết hợp không lường trước được của các primitives được phép khi nguyên tắc được áp dụng rõ ràng.
- Yêu cầu mới giống với yêu cầu bị cấm.
- Các cuộc tấn công kỹ thuật xã hội dựa trên "bạn không nói X bị cấm".

**Bỏ lỡ:**
- Các cuộc tấn công khai thác sự mơ hồ về nguyên tắc ("người dùng yêu cầu điều này nên sự hữu ích nói có").
- Các kịch bản trong đó hai nguyên tắc xung đột theo cách không lường trước được và thứ tự cấp độ không rõ ràng.
- Trôi chậm trong diễn giải nguyên tắc qua training chu kỳ (diễn giải lại).

### Thử nghiệm có sự tham gia năm 2023

Anthropic đã thực hiện một thử nghiệm năm 2023 so sánh hiến pháp do công ty soạn thảo với hiến pháp được tạo ra thông qua ý kiến đóng góp của công chúng (~1,000 người trả lời ở Hoa Kỳ). Hai phiên bản đã đồng ý về ~50% nguyên tắc. Khi chúng khác nhau, phiên bản nguồn công khai hạn chế hơn đối với một số vấn đề (xử lý nội dung chính trị) và ít hạn chế hơn đối với những vấn đề khác (tự tiết lộ danh tính AI). Hiến pháp năm 2026 không kết hợp các phát hiện có nguồn gốc công khai. Đây là một sự căng thẳng được ghi nhận trong cách tiếp cận.

### Tại sao các lệnh cấm được mã hóa cứng là cần thiết

Chỉ riêng alignment dựa trên lý trí không thể khép đuôi. Một kẻ tấn công có thể khiến model chấp nhận một tiền đề (ví dụ: "chúng tôi là một phòng thí nghiệm nghiên cứu vũ khí sinh học được cấp phép") thường có thể nói về các nguyên tắc trong quá khứ phụ thuộc vào lý luận trường hợp. Các lệnh cấm được mã hóa cứng không uốn cong để tạo ra tiền đề. Chúng là Bài 14 "giới hạn hiến pháp cứng" ở lớp alignment.

### Hiến pháp nằm ở đâu trong stack

Hiến pháp không phải là kill switch của Bài 14. Nó sống ở lớp model: trọng lượng của model được huấn luyện để ưa thích. Ngắt kết nối và canary tokens sống ở lớp runtime: những gì runtime cho phép. Cả hai đều là bắt buộc. Một runtime bắn tất cả các hành động sai vì trọng lượng model là cho phép là một vấn đề runtime. Một model từ chối tất cả các hành động đúng đắn vì runtime quá hạn chế là một vấn đề runtime. Các lớp bao phủ các classes khác nhau.

## Ứng dụng

`code/main.py` triển khai trình phân giải mức độ ưu tiên bốn tầng tối thiểu. Người phân giải thực hiện một hành động được đề xuất và một tập hợp các đánh giá nguyên tắc (an toàn, đạo đức, hướng dẫn, tính hữu ích) và trả lại hành động, từ chối hoặc hành động sửa đổi. Người lái xe chạy một bộ trường hợp nhỏ: cho phép rõ ràng, không cho phép rõ ràng, cấm được mã hóa cứng, trường hợp mơ hồ giữa các cấp.

## Sản phẩm bàn giao

`outputs/skill-constitution-review.md` kiểm tra lớp hiến pháp của triển khai: cái gì được mã hóa cứng, cái gì được mã hóa mềm, nơi người vận hành có thể điều chỉnh và liệu hệ thống phân cấp bốn tầng có thực sự là thứ tự giải quyết hay không.

## Bài tập

1. Chạy `code/main.py`. Xác nhận lệnh cấm được mã hóa cứng kích hoạt ngay cả khi tính hữu ích cao. Sửa đổi bộ phân giải để cân nhắc sự hữu ích trên đạo đức; Quan sát chế độ lỗi.

2. Đọc Hiến pháp Claude (công khai, 79 trang, CC0). Xác định một nguyên tắc mà bạn tin là không được xác định. Viết hai đoạn giải thích sự mơ hồ cụ thể và đề xuất một công thức chặt chẽ hơn.

3. Thiết kế bộ mặc định được mã hóa mềm cho agent hỗ trợ khách hàng. Người vận hành điều chỉnh những gì? Người vận hành không thể chạm vào điều gì? Biện minh cho từng ranh giới.

4. Đọc bài báo CAI năm 2022 của Bai và cộng sự. Mô tả một trường hợp mà vòng lặp phê bình và sửa đổi của AI Hiến pháp sẽ tạo ra một kết quả tồi tệ hơn một quy tắc chung. Xác định class.

5. Thử nghiệm có sự tham gia năm 2023 của Anthropic cho thấy sự khác biệt ~50% giữa các nguyên tắc công và công ty. Chọn một danh mục mà điều này quan trọng đối với việc triển khai production (ví dụ: tính trung lập chính trị). Đề xuất một thiết kế cho phép các nhà khai thác thể hiện giá trị của riêng họ trong khi các lệnh cấm được mã hóa cứng vẫn không bị ảnh hưởng.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| AI hiến pháp | "Phương pháp alignment của Anthropic" | Tự phê bình + RLAIF chống lại hiến pháp thành văn |
| alignment dựa trên lý do | "Nguyên tắc, không phải quy tắc" | Model lý do về nguyên tắc xử lý các trường hợp không nhìn thấy |
| Cấm được mã hóa cứng | "Không bao giờ làm X" | Cấm dựa trên quy tắc mà không nhà điều hành hoặc người dùng nào có thể ghi đè |
| Mặc định được mã hóa mềm | "Có thể điều chỉnh bằng người vận hành" | Hành vi trong giới hạn đã khai báo, toán tử kiểm soát |
| Hệ thống phân cấp bốn tầng | "Thứ tự ưu tiên" | An toàn > đạo đức > hướng dẫn > hữu ích |
| RLAIF | "AI phản hồi RL" | RL phần thưởng đến từ những lời phê bình do model tạo ra |
| Hiến pháp có sự tham gia | "Nguyên tắc có nguồn gốc công khai" | Thí nghiệm Anthropic năm 2023; ~50% phân kỳ từ công ty |
| Nguyên tắc trôi dạt | "Phiếu phiên dịch" | Thay đổi chậm trong cách model đọc văn bản nguyên tắc cố định |

## Đọc thêm

- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) - tài liệu CC79 dài 0 trang.
- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) - Bản gốc năm 2022.
- [Anthropic — Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) - thử nghiệm có sự tham gia.
- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) - nơi Hiến pháp nằm trong stack RSP.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) - Vai trò của Hiến pháp trong việc triển khai tầm nhìn dài.
