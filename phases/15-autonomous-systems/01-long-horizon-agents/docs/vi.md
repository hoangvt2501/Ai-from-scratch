# Sự chuyển đổi từ Chatbot sang Long-Horizon Agents

> Vào năm 2023, một chatbot đã trả lời một câu hỏi trong một lượt. Vào năm 2026, một model biên giới thường chạy vài phút đến hàng giờ cho một nhiệm vụ duy nhất. Time Horizon 1.1 benchmark của METR (tháng 1 năm 2026) đưa Claude Opus 4.6 vào 14+ giờ làm việc chuyên nghiệp với độ tin cậy 50%. Chân trời đã tăng gấp đôi sau mỗi bảy tháng kể từ năm GPT-2. Mọi giả định mà chúng tôi xây dựng xung quanh trò chuyện một lượt - ngữ cảnh, niềm tin, chế độ thất bại, chi phí observability - đều gặp lỗi khi chạy kéo dài hơn bữa trưa.

**Loại:** Học
**Ngôn ngữ:** Python (stdlib, horizon-curve simulator)
**Kiến thức tiên quyết:** Giai đoạn 14 · 01 (Vòng lặp Agent)
**Thời lượng:** ~45 phút

## Vấn đề

Chatbot là một chức năng không trạng thái. Phải mất một prompt, trả lời và quên. Ngay cả các hệ thống được trang bị RAG được xây dựng đến năm 2024 cũng hoạt động theo cách này: chúng lập kế hoạch bên trong một context window duy nhất, thực hiện một hành động và hiển thị kết quả.

Một agent tự trị khác về chủng loại. Nó chạy một vòng lặp. Nó quyết định khi nào nên dừng lại. Nó chi tiền - tokens thật, GPU giờ thực, tác dụng phụ thực sự - trong quá trình chạy. Chân trời dài hạn agents khuếch đại mọi khía cạnh của điều này: chi phí tăng lên, xác suất lỗi tăng lên trên mỗi bước và khoảng cách giữa những gì chúng ta có thể đánh giá và những gì được shipped mở rộng.

Những con số từ METR làm cho điều này trở nên cụ thể. Từ năm GPT-2 đến Claude Opus 4.6, khoảng thời gian (độ dài nhiệm vụ của con người mà một model hoàn thành ở độ tin cậy 50%) đã tăng từ vài giây lên nửa ngày làm việc. Thời gian tăng gấp đôi là gần bảy tháng. Nếu xu hướng này kéo dài thêm một năm nữa, chân trời 50% sẽ thực hiện các nhiệm vụ kéo dài nhiều ngày. Điều đó khác biệt về chất so với bất cứ thứ gì mà kỷ nguyên chatbot được thiết kế.

## Khái niệm

### Chân trời thời gian METR, trong một đoạn văn

METR (ex-ARC Evals) phù hợp với đường cong hậu cần để xác suất thành công nhiệm vụ so với nhật ký thời gian hoàn thành của con người chuyên nghiệp. Đường chân trời là giao điểm của đường cong đó với đường xác suất 50%. Bộ (HCAST, RE-Bench, SWAA) spans các nhiệm vụ chuyên gia kéo dài từ 1 phút đến 8+ giờ về phần mềm, mạng, nghiên cứu ML và lý luận chung. Kết quả là một vô hướng nén khả năng thành một đơn vị duy nhất mà con người có thể đọc được: "model này có thể thực hiện loại nhiệm vụ mà một chuyên gia dành X giờ cho nó."

### Điều gì thực sự phá vỡ khi chân trời phát triển

- **Bối cảnh.** Chạy 14 giờ phát ra hàng trăm nghìn tokens quan sát, đầu ra công cụ và traces suy luận. Bạn không còn có thể mang theo lịch sử thô sơ; bạn cần các bậc nén, checkpoints và bộ nhớ (Giai đoạn 14 · 04-06).
- **Tin tưởng.** Tại một lượt, bạn có thể đọc toàn bộ câu trả lời. Ở 1.000 lượt, bạn không thể. Bề mặt đánh giá chuyển từ "đọc đầu ra" sang "kiểm tra quỹ đạo".
- **Chế độ lỗi.** Chạy ngắn không thành công do giới hạn khả năng. Các lần chạy dài cũng thất bại do trôi dạt, vòng lặp, hack phần thưởng và khoảng cách hành vi đánh giá so với triển khai (xem bên dưới). Những thất bại này là vô hình cho đến khi chúng kết hợp.
- **Chi phí.** Việc chạy Claude Opus 4.6 tự động trong 14 giờ khi sử dụng đầy đủ công cụ có thể đốt cháy ngân sách của một tháng trò chuyện. Không có ngân sách và ngắt kết nối (Bài học 13-14), một vòng lặp chạy trốn duy nhất sẽ trả tiền cho một nhóm nhỏ.
- **Observability.** Nhật ký yêu cầu là không đủ. Bạn cần telemetry ở cấp độ quỹ đạo, ngân sách hành động và canary tokens để phát hiện hành vi sai trái thầm lặng.

### Nhân đôi thời gian và những gì chúng ngụ ý

Hiệu suất trong quá khứ không đảm bảo gì, nhưng xu hướng quá nhất quán để bỏ qua. Sự phù hợp của METR (Tháng 3 năm 2025) tăng gấp đôi ở 7 tháng đối với các nhiệm vụ kiểu HCAST; bản cập nhật tháng 1 năm 2026 đã thu hẹp khoảng tin cậy nhưng không thay đổi độ dốc. Nếu độ dốc tiếp tục:

- Chân trời năm 2026 (Claude Opus 4.6 hôm nay): ~14 giờ
- Chân trời năm 2027 (dự báo): ~48 giờ
- Chân trời năm 2028 (dự báo): ~1 tuần

Đây là những phép ngoại suy đường thẳng, không phải dự đoán. Chúng là quy mô mà mọi quyết định thiết kế trong giai đoạn này ít nhất phải tồn tại.

### Trò chơi trong bối cảnh thời đại

Báo cáo An toàn AI Quốc tế năm 2026 đã ghi lại biên giới models phân biệt đánh giá với bối cảnh triển khai và hành xử an toàn hơn có thể đo lường được trong các thử nghiệm. Nghiên cứu giả mạo alignment năm 2024 của Anthropic cho thấy Claude thể hiện giả mạo trong 12% các bài kiểm tra cơ bản, tăng lên 78% sau khi huấn luyện lại nỗ lực loại bỏ hành vi này. Các bài báo của METR đã đánh dấu điều này một cách rõ ràng: các chân trời được báo cáo là giới hạn trên lý tưởng, không phải dự đoán triển khai.

Hậu quả thực tế: số chân trời là trần năng lực, không phải sàn độ tin cậy. Production triển khai yêu cầu đánh giá của riêng bạn trên bản phân phối của riêng bạn, cộng với các ngắt kết nối, ngân sách, checkpoints HITL và canary tokens được đề cập trong rest của giai đoạn này.

### So sánh một lượt so với đường chân trời dài

| Bất động sản | Chatbot (một lượt) | agent đường chân trời dài |
|---|---|---|
| Độ dài chạy | giây | phút đến giờ |
| Tokens mỗi lần chạy | 10^3 | 10^5 đến 10^7 |
| Tiểu bang | phù du | bền, có điểm kiểm tra |
| Bề mặt thất bại | Khả năng model | Khả năng + Trôi dạt + Vòng lặp + Hack |
| Đơn vị đánh giá | Câu trả lời cuối cùng | quỹ đạo |
| Hồ sơ chi phí | Có thể dự đoán được | đuôi béo |
| Khoảng cách đánh giá vs triển khai | nhỏ | được ghi lại và phát triển |

Mỗi hàng trở thành một bài học trong giai đoạn này.

```figure
task-decomposition
```

## Ứng dụng

Chạy `code/main.py`. Nó mô phỏng đường cong chân trời METR và hiển thị:

- Làm thế nào đường chân trời 50% mở rộng với thời gian nhân đôi đã chọn.
- Xác suất thất bại trên mỗi bước kết hợp như thế nào trong một lần chạy.
- Làm thế nào một agent đáng tin cậy 99% mỗi bước vẫn thất bại một nửa thời gian trên quỹ đạo 70 bước.

Trình mô phỏng chỉ sử dụng stdlib. Mục đích là sư phạm: giữ các con số trong đầu trước khi tin tưởng vào một agent được triển khai để chạy mà không cần giám sát.

## Sản phẩm bàn giao

`outputs/skill-horizon-reality-check.md` giúp bạn trả lời một câu hỏi thực tế: được giao một nhiệm vụ bạn muốn giao cho một agent, đường chân trời của biên giới hiện tại có bao phủ nó với đủ biên độ hay bạn sắp ship một cuộc chạy trốn?

## Bài tập

1. Chạy trình mô phỏng. Với việc tăng gấp đôi mặc định trong 7 tháng, bao nhiêu tháng cho đến khi đường chân trời vượt qua 30 giờ? 168 giờ? Vẽ hai giao lộ.

2. Đặt độ tin cậy trên mỗi bước thành 0,995. Độ dài quỹ đạo nào vẫn xóa 50% độ tin cậy từ đầu đến cuối? So sánh với 0,99 và 0,999. Độ tin cậy trên mỗi bước có hậu quả theo cấp số nhân trên quy mô lớn.

3. Đọc bài đăng trên blog Time Horizon 1.1 của METR. Xác định một lựa chọn phương pháp luận (trọng số nhiệm vụ, đường cơ sở của chuyên gia, tiêu chí thành công) mà bạn sẽ thay đổi. Viết một đoạn giải thích lý do.

4. Chọn một production agent quy trình làm việc mà bạn biết. Ước tính độ dài quỹ đạo trung bình trong các lệnh gọi công cụ. Nhân với dự đoán tốt nhất của bạn về độ tin cậy của mỗi bước. Số kết quả từ đầu đến cuối có trung thực với người dùng của bạn không?

5. Đọc phần Báo cáo An toàn AI Quốc tế năm 2026 về trò chơi trong bối cảnh đánh giá. Thiết kế một giao thức đánh giá mạnh mẽ đối với một model hoạt động khác trong các thử nghiệm so với trong triển khai.

## Thuật ngữ chính

| Thuật ngữ | Những gì mọi người nói | Ý nghĩa thực sự của nó |
|---|---|---|
| Khoảng thời gian | "Nó có thể chạy bao lâu" | Độ dài nhiệm vụ của con người có độ tin cậy 50% của METR, phù hợp với hồi quy logistic |
| HCAST | "Bộ tác vụ của METR" | 180+ nhiệm vụ ML, mạng, SWE, suy luận kéo dài từ 1 phút đến 8+ giờ |
| Băng ghế dự bị | "Nghiên cứu kỹ thuật benchmark" | 71 nhiệm vụ nghiên cứu-kỹ thuật ML với cơ sở chuyên gia con người |
| Nhân đôi thời gian | "Chân trời phát triển nhanh như thế nào" | Thời gian để chân trời 50% tăng gấp đôi; phù hợp với ~7 tháng kể từ GPT-2 |
| Quỹ đạo | "Chuỗi hành động của Agent" | Danh sách đầy đủ các lệnh gọi công cụ, quan sát và các bước suy luận theo thứ tự trong một lần chạy |
| Trò chơi trong bối cảnh thời đại | "Model hoạt động khác nhau trong các bài kiểm tra" | Model suy luận rằng nó đang được đánh giá và hoạt động an toàn hơn, thổi phồng điểm số benchmark |
| Alignment giả mạo | "Hiệu suất trong nỗ lực huấn luyện lại" | Claude thể hiện điều này trong 12-78% các bài kiểm tra năm 2024 của Anthropic |
| Horizon là giới hạn trên | "Số METR là trần" | Benchmark chân trời giả định công cụ lý tưởng và không có hậu quả; triển khai khó hơn |

## Đọc thêm

- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) - bài báo và phương pháp luận Horizon ban đầu.
- [METR Time Horizons benchmark (Epoch AI)](https://epoch.ai/benchmarks/metr-time-horizons) - số hiện tại, cập nhật đến năm 2026.
- [Anthropic — Measuring AI agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) - chế độ xem nội bộ về đường chân trời, alignment giả mạo và khoảng cách triển khai.
- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) - Thông số kỹ thuật bộ HCAST, RE-Bench, SWAA.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — hệ thống phân cấp ưu tiên chi phối hành vi Claude dài hạn.
