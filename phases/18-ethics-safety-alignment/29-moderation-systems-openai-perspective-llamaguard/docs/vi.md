# Hệ thống kiểm duyệt - OpenAI, phối cảnh Llama bảo vệ

> Production hệ thống kiểm duyệt vận hành policies an toàn được xác định trong Bài 12-16. OpenAI API kiểm duyệt: `omni-moderation-latest` (2024) được xây dựng trên GPT-4o phân loại văn bản + hình ảnh trong một cuộc gọi; Tốt hơn 42% trên bộ bài kiểm tra đa ngôn ngữ so với phiên bản prior; phản hồi schema trả về 13 danh mục boolean - quấy rối, harassment/threatening, thù hận, hate/threatening, bất hợp pháp, illicit/violent, tự làm hại bản thân, self-harm/intent, self-harm/instructions, tình dục, sexual/minors, bạo lực, violence/graphic; miễn phí cho hầu hết các nhà phát triển. Các mẫu phân lớp: Kiểm duyệt đầu vào (trước khi tạo), Kiểm duyệt đầu ra (sau tạo), Kiểm duyệt tùy chỉnh (quy tắc miền). Các cuộc gọi song song không đồng bộ ẩn độ trễ; phản hồi giữ chỗ trên cờ. Llama Guard 3/4 (Bài 16): 14 mối nguy hiểm MLCommons, Lạm dụng trình thông dịch mã, 8 ngôn ngữ (v3), đa hình ảnh (v4). Perspective API (Google Jigsaw): chấm điểm độc tính trước làn sóng LLM-as-moderator chủ yếu là độc tính một chiều với các biến thể severe-toxicity/insult/profanity; cơ sở cho nghiên cứu kiểm duyệt nội dung. Ngừng sử dụng: Azure Content Moderator không dùng nữa vào tháng 2 năm 2024, ngừng hoạt động vào tháng 2 năm 2027, được thay thế bằng Azure AI Content Safety.

**Loại:** Xây dựng
**Ngôn ngữ:** Python (stdlib, three-layer moderation harness)
**Kiến thức tiên quyết:** Giai đoạn 18 · 16 (Llama Bảo vệ / Garak / PyRIT)
**Thời lượng:** ~60 phút

## Mục tiêu học tập

- Mô tả phân loại danh mục của API OpenAI Moderation và nó khác với bộ MLCommons của Llama Guard 3 như thế nào.
- Mô tả ba mẫu lớp kiểm duyệt (đầu vào, đầu ra, tùy chỉnh) và đặt tên cho một chế độ lỗi của mỗi loại.
- Mô tả vị trí của Perspective API như một đường cơ sở trước thời kỳ LLM và lý do tại sao nó vẫn được sử dụng trong nghiên cứu.
- Nêu dòng thời gian ngừng sử dụng Azure.

## Vấn đề

Bài 12-16 mô tả các cuộc tấn công và công cụ phòng thủ. Bài 29 bao gồm các hệ thống kiểm duyệt đã triển khai để vận hành các biện pháp phòng thủ ở bề mặt nơi người dùng chạm vào sản phẩm. Mô hình ba lớp là configuration mặc định năm 2026.

## Khái niệm

### OpenAI API kiểm duyệt

`omni-moderation-latest` (2024). Được xây dựng trên GPT-4o. Phân loại văn bản + hình ảnh trong một cuộc gọi. Miễn phí cho hầu hết các nhà phát triển.

Thể loại (13 boolean trong phản hồi schema):
- quấy rối, harassment/threatening
- ghét bỏ, hate/threatening
- tự làm hại bản thân, self-harm/intent, self-harm/instructions
- tình dục, sexual/minors
- bạo lực, violence/graphic
- bất hợp pháp, illicit/violent

Hỗ trợ đa phương thức áp dụng cho `violence`, `self-harm` và `sexual` nhưng không áp dụng `sexual/minors`; Các rest chỉ có văn bản.

Đối với mã harness trong `code/main.py`, chúng tôi thu gọn các danh mục phụ `/threatening`, `/intent`, `/instructions` và `/graphic` thành các thể loại phụ cấp cao nhất của chúng để đơn giản hóa sư phạm. Mã Production nên sử dụng schema đầy đủ gồm 13 danh mục.

Tốt hơn 42% trên bộ kiểm tra đa ngôn ngữ so với endpoint kiểm duyệt thế hệ prior. Điểm cho mỗi danh mục; ứng dụng đặt ngưỡng.

### Llama Bảo vệ 3/4

Được đề cập trong Bài 16. 14 loại nguy hiểm MLCommons (được tổ chức khác với 13 boolean schema phản hồi của OpenAI). Hỗ trợ 8 ngôn ngữ (v3). Llama Guard 4 (tháng 4 năm 2025) nguyên bản là đa phương thức, 12B.

Phân loại OpenAI và Llama Vệ binh trùng lặp nhưng khác nhau. OpenAI có "bất hợp pháp" như một phạm trù rộng; Llama Guard có "tội phạm bạo lực" và "tội phạm phi bạo lực" riêng biệt. Triển khai chọn dựa trên sự phù hợp policy phân loại của chúng.

### Phối cảnh API (Google Jigsaw)

Hệ thống chấm điểm độc tính trước làn sóng LLM làm người kiểm duyệt (trước năm 2020). Danh mục: ĐỘC HẠI, SEVERE_TOXICITY, XÚC PHẠM, TỤC TĨU, ĐE DỌA, IDENTITY_ATTACK. Điểm chính một chiều (TOXICITY) với các biến thể thứ nguyên phụ.

Được sử dụng rộng rãi làm cơ sở nghiên cứu kiểm duyệt nội dung vì API ổn định, được ghi lại và có nhiều năm dữ liệu hiệu chuẩn. Đối với các trường hợp sử dụng hiện đại LLM liền kề, Llama Guard hoặc OpenAI Moderation thường phù hợp hơn.

### Mô hình ba lớp

1. **Kiểm duyệt đầu vào.** Phân loại prompt của người dùng trước khi tạo. Từ chối nếu bị gắn cờ. Độ trễ: một cuộc gọi phân loại.
2. **Kiểm duyệt đầu ra.** Phân loại đầu ra của model trước khi giao hàng. Thay thế bằng từ chối nếu bị gắn cờ. Độ trễ: một cuộc gọi phân loại sau thế hệ.
3. **Kiểm duyệt tùy chỉnh.** Quy tắc dành riêng cho miền (biểu thức chính quy, danh sách cho phép, policy kinh doanh). Chạy ở đầu vào hoặc đầu ra.

Ba lớp được thiết kế tuần tự: kiểm duyệt đầu vào phải hoàn thành trước khi tạo và kiểm duyệt đầu ra chạy sau khi tạo. Tính song song áp dụng trong một lớp — chạy đồng thời nhiều bộ phân loại (ví dụ: OpenAI Kiểm duyệt + Bảo vệ Llama + Phối cảnh) trên cùng một văn bản sẽ ẩn độ trễ của mỗi bộ phân loại. Là một tối ưu hóa tùy chọn, phản hồi giữ chỗ ("một khoảnh khắc, kiểm tra...") có thể được hiển thị trong khi kiểm duyệt đầu vào hoàn tất và streaming token-1 bị trì hoãn. Hành vi gắn cờ có thể định cấu hình: từ chối, dọn dẹp, chuyển sang xem xét của con người.

### Chế độ thất bại

- **Chỉ đầu vào.** Không bắt ảo giác đầu ra (Bài 12-14 các cuộc tấn công mã hóa bỏ qua bộ phân loại đầu vào).
- **Chỉ đầu ra.** Cho phép bất kỳ đầu vào nào đến model; tăng chi phí; Hiển thị lý luận nội bộ cho kẻ tấn công.
- **Chỉ tùy chỉnh.** Không mạnh mẽ trên các danh mục; Biểu thức chính quy rất giòn.

Phân lớp là mặc định. Thắt lưng và dây treo.

### Ngừng sử dụng Azure

Azure Content Moderator: không dùng nữa vào tháng 2 năm 2024, ngừng hoạt động vào tháng 2 năm 2027. Được thay thế bằng Azure AI An toàn nội dung, dựa trên LLM và tích hợp với Azure OpenAI. Việc di chuyển là một dự án cấp trường 2024-2027 để triển khai Azure.

### Điều này phù hợp với Giai đoạn 18 ở đâu

Bài 16 đề cập đến công cụ kiểm duyệt trong bối cảnh đội đỏ. Bài 29 bao gồm kiểm duyệt được triển khai. Bài 30 kết thúc với bằng chứng về khả năng sử dụng kép hiện tại.

## Ứng dụng

`code/main.py` xây dựng một harness kiểm duyệt ba lớp: người kiểm duyệt đầu vào (từ khóa + điểm danh mục), người kiểm duyệt đầu ra (cùng một bộ phân loại trên đầu ra), người kiểm duyệt tùy chỉnh (quy tắc miền). Bạn có thể chạy đầu vào và quan sát lớp nào bắt được nội dung.

## Sản phẩm bàn giao

Bài học này tạo ra `outputs/skill-moderation-stack.md`. Với một triển khai, nó đề xuất một stack configuration kiểm duyệt: bộ phân loại nào ở đầu vào, cái nào ở đầu ra, quy tắc tùy chỉnh nào và cái nào đánh giá cho các trường hợp biên.

## Bài tập

1. Chạy `code/main.py`. Chạy đầu vào lành tính, ranh giới và có hại qua cả ba lớp. Báo cáo lớp nào kích hoạt cho từng lớp.

2. Mở rộng harness bằng cách chấm điểm độc tính theo kiểu API phối cảnh trên một danh mục cụ thể. So sánh hành vi ngưỡng của nó với điểm danh mục.

3. Đọc tài liệu API Kiểm duyệt OpenAI và danh sách danh mục Llama Guard 3. Ánh xạ từng danh mục OpenAI với các danh mục Vệ binh Llama gần nhất. Xác định ba danh mục không được lập bản đồ rõ ràng.

4. Thiết kế stack kiểm duyệt để triển khai trợ lý mã (ví dụ: GitHub Copilot). Xác định các danh mục liên quan nhất và ít liên quan nhất và đề xuất các quy tắc tùy chỉnh.

5. Azure Content Moderator ngừng hoạt động vào tháng 2 năm 2027. Lập kế hoạch di chuyển sang Azure AI An toàn nội dung. Xác định yếu tố có nguy cơ cao nhất của việc di chuyển.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|------|-----------------|------------------------|
| OpenAI Kiểm duyệt | "Omni-moderation-mới nhất" | Bộ phân loại 13 danh mục (văn bản) dựa trên GPT-4o với hỗ trợ đa phương thức một phần |
| Phối cảnh API | "Độc tính của Google Jigsaw" | Đường cơ sở chấm điểm độc tính trước thời kỳ LLM |
| Bảo vệ Llama | "MLCommons 14 hạng mục" | Công cụ phân loại rủi ro của Meta (v3: 8B văn bản, 8 lang; v4: 12B đa phương thức) |
| Kiểm duyệt đầu vào | "Bộ lọc tiền thế hệ" | Bộ phân loại trên prompt người dùng trước model gọi |
| Kiểm duyệt đầu ra | "Bộ lọc sau thế hệ" | Phân loại trên đầu ra model trước khi giao hàng |
| Kiểm duyệt tùy chỉnh | "Quy tắc miền" | Quy tắc dành riêng cho việc triển khai (biểu thức chính quy, danh sách cho phép, policy) |
| Kiểm duyệt theo lớp | "cả ba lớp" | Mẫu triển khai production tiêu chuẩn |

## Đọc thêm

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — endpoint kiểm duyệt toàn diện
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Vệ binh repo
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) - Chấm điểm độc tính
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Thay thế Azure
