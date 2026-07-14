# Anthropic Mở rộng quy mô có trách nhiệm Policy v3.0

> RSP v3.0 có hiệu lực từ ngày 24 tháng 2 năm 2026, thay thế policy 2023. Giảm thiểu hai tầng: những gì Anthropic sẽ đơn phương làm so với những gì được đóng khung như một khuyến nghị toàn ngành (bao gồm các tiêu chuẩn bảo mật RAND SL-4). Thêm Lộ trình An toàn Biên giới và Báo cáo Rủi ro dưới dạng tài liệu thường trực thay vì sản phẩm phân phối một lần. Bỏ cam kết tạm dừng năm 2023. Giới thiệu ngưỡng R&D-4 AI: một khi vượt qua, Anthropic phải công bố một trường hợp khẳng định xác định các rủi ro và giảm thiểu sai lệch. Claude Opus 4.6 không vượt qua nó. Anthropic tuyên bố trong thông báo v3.0 rằng "tự tin loại trừ điều này đang trở nên khó khăn". SaferAI đánh giá RSP năm 2023 ở mức 2,2; họ đã hạ cấp v3.0 xuống 1.9, đưa Anthropic vào danh mục RSP "yếu" cùng với OpenAI và DeepMind. Ngưỡng định tính thay thế các cam kết định lượng năm 2023; Loại bỏ mệnh đề tạm dừng là hồi quy rõ ràng nhất.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, RSP threshold decision engine)
**Kiến thức tiên quyết:** Giai đoạn 15 · 06 (AAR), Giai đoạn 15 · 07 (RSI)
**Thời lượng:** ~45 phút

## Vấn đề

Các phòng thí nghiệm biên giới công bố các policies mở rộng quy mô một phần là tài liệu kỹ thuật, một phần là tài liệu quản trị và một phần tín hiệu cho các cơ quan quản lý. RSP v3.0 là tài liệu Anthropic hiện tại. Đọc kỹ nó quan trọng không phải vì việc tuân thủ nó có tính ràng buộc (nó không phải vậy), mà vì khung định hình cách một phòng thí nghiệm quan niệm về rủi ro thảm khốc và cách họ truyền đạt sự đánh đổi cho công chúng.

Sự khác biệt v3.0 và v2.0 là đơn vị hữu ích. Những gì đã được thêm vào: Lộ trình An toàn Biên giới, Báo cáo Rủi ro, ngưỡng R&D-4 AI. Những gì đã bị xóa: cam kết tạm dừng năm 2023. Những gì đã được định hình lại: một lịch trình giảm thiểu hai cấp được phân chia giữa Anthropic đơn phương và khuyến nghị của ngành. Đánh giá bên ngoài - SaferAI - đã hạ điểm từ 2.2 (v2) xuống 1.9 (v3.0). Đây là cách một policy cạo có thể trở nên ít nghiêm ngặt hơn trong khi trông bóng bẩy hơn.

## Khái niệm

### Lịch trình giảm thiểu hai cấp

- **Anthropic Hành động đơn phương**: Anthropic sẽ làm gì bất kể các phòng thí nghiệm khác làm gì. Training dừng trên ngưỡng, các biện pháp bảo mật cụ thể, cổng triển khai cụ thể.
- **Khuyến nghị toàn ngành**: những gì Anthropic nghĩ ngành nên làm chung. Bao gồm các tiêu chuẩn bảo mật RAND SL-4. Đây không phải là những cam kết của Anthropic; họ policy vận động.

Cấu trúc hai tầng không có trong v2. Nó có nghĩa là người đọc cần xem mỗi cam kết nằm trong cột nào. Một biện pháp bảo mật trong cột "khuyến nghị toàn ngành" không phải là lời hứa của Anthropic; đó là hy vọng của Anthropic.

### Ngưỡng R&D-4 AI

Đây là tên cấp độ khả năng RSP v3.0 là ngưỡng quan trọng tiếp theo. Cụ thể: một model có thể tự động hóa một phần đáng kể nghiên cứu AI với chi phí cạnh tranh. Một khi Anthropic tin rằng một model vượt qua nó, họ phải công bố một trường hợp khẳng định xác định các rủi ro sai lệch và giảm thiểu trước khi tiếp tục mở rộng quy mô.

Claude Opus 4.6 không vượt qua nó theo thông báo v3.0. Tài liệu nói thêm: "tự tin loại trừ điều này đang trở nên khó khăn." Cụm từ đó rất quan trọng; Nó thừa nhận rằng ngưỡng đủ gần để trở thành một mối quan tâm trực tiếp, không phải là một giới hạn đầu cơ.

Bài 6 (Nghiên cứu Alignment tự động) và Bài 7 (Tự cải thiện đệ quy) đưa trực tiếp vào ngưỡng này. Các nhà nghiên cứu alignment tự động vượt qua các thanh chất lượng nghiên cứu là bằng chứng cho thấy ngưỡng AI R&D-4 đang đến gần.

### Lộ trình an toàn biên giới và báo cáo rủi ro

Phiên bản 3.0 nâng hai loại artifact thành tài liệu thường trực:

- **Lộ trình An toàn Biên giới**: tài liệu hướng tới tương lai mô tả công việc an toàn theo kế hoạch, kỳ vọng về khả năng và nghiên cứu giảm thiểu.
- **Báo cáo rủi ro**: tài liệu hồi cứu về models cụ thể sau khi phát hành, mô tả khả năng quan sát được và rủi ro còn lại.

Cả hai đều công khai. Cả hai đều được cập nhật theo nhịp đã công bố. Tiện ích là: người đọc có thể theo dõi những gì Anthropic nói họ sẽ làm trong Lộ trình so với những gì họ báo cáo trong Báo cáo rủi ro.

### Xóa điều khoản tạm dừng

RSP năm 2023 bao gồm cam kết tạm dừng rõ ràng: nếu một model vượt qua ngưỡng năng lực cụ thể, training sẽ tạm dừng cho đến khi có biện pháp giảm thiểu. V3.0 thay thế tạm dừng rõ ràng bằng một công thức nhẹ nhàng hơn (công bố một trường hợp khẳng định, tiến hành nếu các biện pháp giảm thiểu là đủ). SaferAI và các nhà phân tích khác trực tiếp gọi đây là hồi quy mạnh nhất trong tài liệu mới.

Lập luận policy cho sự thay đổi: các ngưỡng định lượng vào năm 2023 hóa ra không thể đạt được bởi khả năng của năm 2026 benchmarks vì bản thân các benchmarks đã được mở rộng lại. Lập luận phản đối: điều khoản tạm dừng trong policy chia tỷ lệ là một thiết bị cam kết; Loại bỏ nó sẽ làm mất uy tín của policy.

### Hạ cấp của SaferAI

SaferAI là một tổ chức độc lập đánh giá các tài liệu kiểu RSP. Xếp hạng công khai của họ: RSP năm 2023 Anthropic đạt 2,2 điểm (trong thang điểm mà 4,0 là RSP tốt nhất hiện tại và 1,0 là danh nghĩa). v3.0 đạt 1.9 điểm. Điều này đã chuyển Anthropic từ "trung bình" sang "yếu", tham gia OpenAI và DeepMind trong danh mục yếu.

Các yếu tố hạ cấp cho mỗi SaferAI:
- Ngưỡng định tính thay thế ngưỡng định lượng.
- Đã xóa cam kết tạm dừng.
- AI giảm thiểu ngưỡng R&D-4 được mô tả là "trường hợp khẳng định" chứ không phải là các biện pháp cụ thể.
- Các cơ chế đánh giá phụ thuộc vào Nhóm Tư vấn An toàn của Anthropic, với sự giám sát độc lập hạn chế.

### Bài học này không phải là gì

Đây không phải là một bài học về sự tuân thủ. RSP v3.0 không phải là một quy định; không có gì buộc Anthropic phải tuân theo nó. Bài học là đọc tài liệu với tính cụ thể và hoài nghi mà nó xứng đáng. Mở rộng quy mô policies là tín hiệu công cộng chính mà các phòng thí nghiệm biên giới phát ra về tình trạng rủi ro thảm họa. Đọc kỹ chúng là một skill thiết thực cho bất kỳ ai có công việc phụ thuộc vào khả năng biên giới.

## Ứng dụng

`code/main.py` triển khai một công cụ quyết định nhỏ phản ánh hình dạng đánh giá ngưỡng RSP: đưa ra một model ứng cử viên và một tập hợp các phép đo khả năng, trả về liệu ngưỡng R&D-4 AI có bị vượt qua hay không, các phần trường hợp khẳng định bắt buộc và liệu việc triển khai có thể tiến hành hay không. Nó đơn giản một cách có chủ đích; Vấn đề là làm cho logic của tài liệu trở nên rõ ràng.

## Sản phẩm bàn giao

`outputs/skill-scaling-policy-review.md` xem xét policy mở rộng quy mô (Anthropic, OpenAI, DeepMind hoặc nội bộ) so với tham chiếu phiên bản 3.0: cấu trúc hai cấp, ngưỡng, cam kết tạm dừng, đánh giá độc lập.

## Bài tập

1. Chạy `code/main.py`. Cho ăn trong ba models tổng hợp ở các mức khả năng khác nhau. Xác nhận trình đánh giá ngưỡng hoạt động như mong đợi và tạo mẫu trường hợp khẳng định phù hợp.

2. Đọc đầy đủ RSP v3.0 (32 trang). Xác định mọi cam kết nằm trong cấp "đề xuất toàn ngành". Cam kết nào trong số những cam kết đó sẽ là "Anthropic đơn phương" trong v2?

3. Đọc phương pháp phân loại RSP của SaferAI. Tái tạo điểm 1.9 của họ cho v3.0 bằng cách áp dụng bảng đánh giá của họ vào tài liệu. Hàng bảng đánh giá nào dẫn đến việc hạ cấp nhiều nhất?

4. Cam kết tạm dừng năm 2023 đã bị loại bỏ. Đề xuất một cam kết thay thế để duy trì uy tín của policy đồng thời thừa nhận vấn đề thay đổi quy mô benchmark năm 2026.

5. So sánh RSP phiên bản 3.0 với Chuẩn bị sẵn sàng OpenAI Framework phiên bản 2 (Bài 20). Chọn một khu vực mà phiên bản 3.0 mạnh hơn. Chọn một khu vực mà Framework Chuẩn bị mạnh hơn.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| RSP | "Anthropic mở rộng quy mô policy" | Policy mở rộng quy mô có trách nhiệm; phiên bản 3.0 có hiệu lực từ ngày 24 tháng 2 năm 2026 |
| AI Nghiên cứu và Phát triển-4 | "Ngưỡng tự động hóa nghiên cứu" | Khả năng tự động hóa nghiên cứu AI đáng kể với chi phí cạnh tranh |
| Trường hợp khẳng định | "Biện minh an toàn" | Lập luận được công bố rằng các rủi ro được xác định và giảm thiểu đầy đủ |
| Lộ trình an toàn biên giới | "Kế hoạch phía trước" | Tài liệu thường trực về công tác an toàn theo kế hoạch và năng lực dự kiến |
| Báo cáo rủi ro | "Hồi tưởng trên một model" | Tài liệu thường trực về khả năng quan sát được và rủi ro còn lại sau khi phát hành |
| Giảm thiểu hai cấp | "Đơn phương vs công nghiệp" | Anthropic cam kết so với khuyến nghị của ngành, tách biệt |
| Cam kết tạm dừng | "Điều khoản 2023" | Lời hứa rõ ràng sẽ tạm dừng training; Đã xóa trong v3.0 |
| Xếp hạng SaferAI | "Cấp RSP độc lập" | Bảng đánh giá của bên thứ ba; v3.0 đạt 1.9 điểm (v2 là 2.2) |

## Đọc thêm

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) - policy đầy đủ 32 trang.
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — tóm tắt các thay đổi từ phiên bản 2.
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — tài liệu thường trực được liên kết từ RSP v3.0.
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) - hồi tưởng về biên giới hiện tại model.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — kết nối AI R&D-4 với quyền tự chủ đo lường.
