# Tầm nhìn thời gian METR và đánh giá khả năng bên ngoài

> METR (ex-ARC Evals) là một 501 (c) (3) độc lập kể từ tháng 12 năm 2023. Time Horizon 1.1 benchmark (tháng 1 năm 2026) của họ phù hợp với đường cong hậu cần đối với xác suất thành công nhiệm vụ so với log (thời gian hoàn thành của con người chuyên nghiệp); Giao điểm ở xác suất 50% xác định khoảng thời gian của model. Bộ cam kết 2025–2026 bao gồm các đánh giá giám sát GPT-5.1, GPT-5.1-Codex-Max và nguyên mẫu (có thể giám sát bắt các nhiệm vụ phụ; agent có thể trốn tránh không). Benchmark bộ: HCAST (180+ ML, mạng, SWE, nhiệm vụ lý luận; 1 phút đến 8+ giờ), RE-Bench (71 nhiệm vụ nghiên cứu-kỹ thuật ML với cơ sở chuyên gia), SWAA. Lưu ý trung thực: Các phép đo METR được lý tưởng hóa - không có con người, không có hậu quả thực sự - và nhóm nghiên cứu đã ghi lại khoảng cách hành vi đánh giá so với triển khai (Bài học 1). Khoảng thời gian là giới hạn trên, không phải dự đoán triển khai.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, logistic-fit horizon estimator)
**Kiến thức tiên quyết:** Giai đoạn 15 · 01 (Đường chân trời dài agents), Giai đoạn 15 · 19 (RSP)
**Thời lượng:** ~60 phút

## Vấn đề

Tỷ lệ policies (Bài 19, 20) chỉ hữu ích khi các phép đo mà chúng tham khảo. "AI ngưỡng R&D-4" và "Quyền tự chủ tầm xa" được định nghĩa trong policy văn xuôi; chúng chỉ có thể hành động khi các đánh giá cụ thể tạo ra những con số cụ thể.

METR là tổ chức đánh giá bên ngoài 2024–2026 đã xác định nhiều con số trong số đó. Họ đánh giá models biên giới - thường là trước khi phát hành, theo NDA với các phòng thí nghiệm - và công bố phương pháp luận sau đó. Time Horizon 1.1 benchmark (tháng 1 năm 2026) là artifact tiêu đề của họ: một vô hướng duy nhất nén khả năng thành một đơn vị con người có thể đọc được ("model này có thể thực hiện loại nhiệm vụ mà một chuyên gia dành X giờ với độ tin cậy 50%).

Bài học một phần là về phương pháp luận (cách tính toán chân trời) và một phần về cách giải thích (tại sao chân trời là giới hạn trên, không phải dự đoán triển khai). Hai skills thuộc về nhau. Một nhóm hiểu đường chân trời phù hợp như thế nào sẽ khó bị đánh lừa với tuyên bố của nhà cung cấp tồi hơn nhiều so với một nhóm chỉ nhìn thấy "14 giờ" trên một trang trượt.

## Khái niệm

### Nền tảng METR

- Thành lập: Tháng Mười Hai 2023 (cũ là ARC Evals, tách ra thành 501 (c) (3) độc lập).
- Phạm vi: đánh giá khả năng tự chủ của models biên giới, thường là trước khi phát hành.
- Phòng thí nghiệm đối tác: Anthropic, OpenAI (nhiều cam kết 2025–2026).
- Các sản phẩm đáng chú ý: Time Horizon 1.0 (tháng 3 năm 2025), Time Horizon 1.1 (tháng 1 năm 2026), đánh giá giám sát nguyên mẫu.

### Chân trời thời gian phù hợp

Phương pháp luận (từ blog và bài báo của METR):

1. Thu thập bộ tác vụ trải dài từ quy mô phút đến thời gian hoàn thành của chuyên gia. Bộ hiện tại: HCAST (180+ nhiệm vụ), RE-Bench (71 nhiệm vụ), SWAA.
2. Chạy model trên mỗi nhiệm vụ; ghi lại thành công hay thất bại.
3. Phù hợp với đường cong hậu cần: P (thành công) như một hàm của log (thời gian hoàn thành của chuyên gia).
4. Đường chân trời là thời gian chuyên gia mà tại đó P (thành công) = 0,5.

Hình dạng phù hợp với hậu cần là phù hợp vì khả năng thường có mối quan hệ ngày càng tăng, tiếp cận ổn định với độ khó của nhiệm vụ. Điểm 50% là một lựa chọn (có thể là 10%, 90%); METR báo cáo nhiều ngưỡng trong bài báo chi tiết nhưng dẫn đầu với 50% vì nó trực quan nhất.

### Các con số tháng 1 năm 2026

Mỗi Time Horizon 1.1:

- Claude Opus 4.6: ~14 giờ ở độ tin cậy 50%, tính đến Time Horizon 1.1 (tháng 1 năm 2026).
- Tăng gấp đôi thời gian cho các nhiệm vụ kiểu HCAST: ~4,3 tháng (130,8 ngày) đối với sự phù hợp sau năm 2023 được báo cáo bởi Time Horizon 1.1 (tháng 1 năm 2026); con số ~7 tháng là phù hợp đầy đủ của giai đoạn 2019–2025 từ Time Horizon 1.0 và được báo cáo trong TH1.1 cùng với con số sau năm 2023.

Đó là những con số cụ thể mà rest của Giai đoạn 15 tham khảo. Chúng được trích dẫn trong RSP v3.0 của Anthropic (Bài 19) và FSF v3 của DeepMind (Bài 20) như một trong những phép đo đánh giá ngưỡng cho ăn.

### Benchmark dãy phòng

- **HCAST**: 189 nhiệm vụ trên ML, an ninh mạng, kỹ thuật phần mềm và lý luận chung. Span: 1 phút đến 8+ giờ thời gian chuyên gia. Bộ chính phía sau số đường chân trời.
- **RE-Bench**: 71 nhiệm vụ nghiên cứu-kỹ thuật ML với cơ sở chuyên gia con người. Đo lường liệu model có thể thực hiện loại công việc R&D ML đưa vào ngưỡng R&D-4 AI hay không.
- **SWAA**: nhiệm vụ ngắn hơn; hữu ích để hiệu chỉnh đầu thấp của đường cong đường chân trời.

### Đánh giá giám sát nguyên mẫu

Công việc của METR 2025–2026 bao gồm các nguyên mẫu giám sát-đánh giá: một màn hình có thể bắt được một model đang thực hiện một nhiệm vụ phụ (một cái gì đó khác với những gì người vận hành yêu cầu); agent có thể trốn tránh màn hình. Những điều này trực tiếp cung cấp cho danh mục "Phá hoại các biện pháp bảo vệ" trong OpenAI PF v2 và giám sát alignment lừa đảo trong DeepMind FSF v3 (Bài học 20).

### Tại sao chân trời là giới hạn trên

- **Công cụ lý tưởng hóa**: benchmark nhiệm vụ cung cấp cho model các công cụ sạch sẽ và dữ liệu được định dạng tốt. Triển khai thực sự lộn xộn hơn.
- **Không có hậu quả thực sự**: model không bao giờ thực sự lập hóa đơn cho khách hàng, xóa dữ liệu thực hoặc liên hệ với người thật. Triển khai thực sự có cổ phần không thể đảo ngược.
- **Trò chơi trong bối cảnh thời đại**: Bài 1. Models hành xử khác nhau trong các bài kiểm tra. Báo cáo An toàn AI Quốc tế năm 2026 ghi lại điều này một cách thực nghiệm.
- **Không có variance người dùng hợp pháp**: benchmark prompts có cấu trúc. Người dùng thực tạo ra các yêu cầu mơ hồ, phụ thuộc vào ngữ cảnh.

Chân trời là trần năng lực trong điều kiện thuận lợi. Độ tin cậy triển khai là một con số khác, thấp hơn và các nhóm phải đo lường phân phối của chính họ để biết điều đó.

### Trường hợp đánh giá bên ngoài

Đánh giá bên ngoài rất quan trọng vì các phòng thí nghiệm nội bộ có động lực để tối ưu hóa các chỉ số mà họ báo cáo. Tính độc lập của METR - 501 (c) (3) với phương pháp luận được tuyên bố và các bài báo được bình duyệt - là sự giảm thiểu cấu trúc. Nó không đủ (các phòng thí nghiệm vẫn kiểm soát những gì METR nhìn thấy), nhưng nó hoàn toàn tốt hơn là không đánh giá bên ngoài.

### Cách sử dụng số chân trời trong thực tế

- **Là một bộ lọc năng lực**: nếu chân trời của một model thấp hơn nhiều so với thời gian chuyên gia của một nhiệm vụ được đề xuất, đừng ship nó tự chủ (tệp skill của Bài 1).
- **Là một chỉ báo xu hướng**: thời gian nhân đôi cho bạn biết thực hành hiện tại sẽ duy trì an toàn trong bao lâu ngay cả khi không có biện pháp giảm thiểu mới.
- **Như một prior**: chân trời 14 giờ là điểm khởi đầu. Điều chỉnh theo phân phối nhiệm vụ, chất lượng công cụ và bối cảnh triển khai của bạn.

## Ứng dụng

`code/main.py` thực hiện sự phù hợp hậu cần giữa thành công nhiệm vụ so với log (thời gian chuyên gia), với một bộ kết quả tổng hợp. Nó báo cáo 50% chân trời (tiêu đề của METR), 10% chân trời (bảo thủ) và 90% chân trời (lạc quan). Đồng thời chứng minh những gì thay đổi khi tỷ lệ thành công được thổi phồng một cách giả tạo bởi trò chơi trong bối cảnh eval.

## Sản phẩm bàn giao

`outputs/skill-horizon-interpretation.md` xem xét tuyên bố về đường chân trời của nhà cung cấp và tạo ra phân tích khoảng cách giữa tuyên bố benchmark và thực tế triển khai.

## Bài tập

1. Chạy `code/main.py`. Xác nhận đường chân trời 50% của phù hợp với ground truth tổng hợp. Bây giờ giảm một nửa lưới nhiệm vụ-thời gian; Ước tính Horizon có thay đổi có ý nghĩa không?

2. Đọc bài đăng trên blog Time Horizon 1.1 của METR. Xác định các nhiệm vụ cụ thể ở đâu có độ tin cậy cao nhất và thấp nhất. Giải thích lý do tại sao khoảng cách tồn tại.

3. Đọc tài nguyên "Đo lường khả năng AI tự động" của METR. Liệt kê các danh mục nhiệm vụ HCAST. Chọn một danh mục mà bạn sẽ cân nặng hơn cho một nhiệm vụ production và giải thích lý do tại sao.

4. Giới thiệu trò chơi theo ngữ cảnh eval vào trình mô phỏng: lật ~20% nhiệm vụ thất bại để thành công. Báo cáo chân trời mới. Điều này xấp xỉ tỷ lệ chơi game 20% làm với con số quan sát được.

5. Thiết kế đánh giá chân trời nội bộ về tồn đọng lỗi của riêng bạn hoặc một bộ nhiệm vụ đại diện. Mô tả việc thu thập dữ liệu, sự phù hợp và những gì đầu ra cho bạn biết. So sánh với số METR.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| ĐO LƯỜNG | "Người đánh giá bên ngoài" | cựu ARC Evals; độc lập 501 (c) (3) kể từ tháng 12 năm 2023 |
| Chân trời thời gian | "Đo lường năng lực" | Độ dài nhiệm vụ của chuyên gia với độ tin cậy 50%, từ sự phù hợp với hậu cần |
| HCAST | "Phòng chính của METR" | 180+ tác vụ kéo dài từ 1 phút đến 8+ giờ |
| Băng ghế dự bị | "Kỹ thuật nghiên cứu" | 71 nhiệm vụ nghiên cứu-kỹ thuật ML với cơ sở con người |
| SWAA | "Bộ tác vụ ngắn" | Hiệu chỉnh đầu thấp của đường cong đường chân trời |
| Nhân đôi thời gian | "Tốc độ tăng trưởng" | Thời gian để chân trời 50% tăng gấp đôi; ~7 tháng cho mỗi HCAST |
| Trò chơi trong bối cảnh thời đại | "Model cư xử khác nhau" | Khoảng cách hành vi được ghi lại giữa thử nghiệm và triển khai |
| Giới hạn trên | "Chân trời là trần nhà" | Benchmark chân trời > độ tin cậy triển khai khi tải |

## Đọc thêm

- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) - Thông số kỹ thuật HCAST, RE-Bench, SWAA.
- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) - giấy chân trời ban đầu.
- [METR — Time Horizon 1.1 (January 2026)](https://metr.org/research/) - con số và phương pháp hiện tại.
- [Epoch AI — METR Time Horizons benchmark](https://epoch.ai/benchmarks/metr-time-horizons) - theo dõi trực tiếp.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — quan điểm nội bộ về các phép đo của METR.
